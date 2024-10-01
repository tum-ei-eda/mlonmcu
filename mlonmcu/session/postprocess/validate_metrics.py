#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Validation metrics utilities."""

import ast
import numpy as np
from typing import Optional

from mlonmcu.logging import get_logger

logger = get_logger()


class ValidationMetric:
    def __init__(self, name, **cfg):
        self.name = name
        self.num_total = 0
        self.num_correct = 0

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        raise NotImplementedError

    def check(self, out_data, out_data_ref, quant: bool = False):
        return out_data.dtype == out_data_ref.dtype

    def process(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        if not self.check(out_data, out_data_ref, quant=quant):
            return
        self.num_total += 1
        if self.process_(out_data, out_data_ref):
            self.num_correct += 1

    def get_summary(self):
        if self.num_total == 0:
            return "N/A"
        return f"{self.num_correct}/{self.num_total} ({int(self.num_correct/self.num_total*100)}%)"


class ClassifyMetric:
    def __init__(self, name, **cfg):
        self.name = name
        self.num_total = 0
        self.num_correct = 0

    def process_(self, out_data, label_ref, quant: bool = False):
        raise NotImplementedError

    def check(self, out_data, label_ref, quant: bool = False):
        return True

    def process(self, out_data, label_ref, quant: bool = False):
        if not self.check(out_data, label_ref, quant=quant):
            return
        self.num_total += 1
        if self.process_(out_data, label_ref):
            self.num_correct += 1

    def get_summary(self):
        if self.num_total == 0:
            return "N/A"
        return f"{self.num_correct}/{self.num_total} ({int(self.num_correct/self.num_total*100)}%)"


class AllCloseMetric(ValidationMetric):
    def __init__(self, name: str, atol: float = 0.0, rtol: float = 0.0):
        super().__init__(name)
        assert atol >= 0
        self.atol = atol
        assert rtol >= 0
        self.rtol = rtol

    def check(self, out_data, out_data_ref, quant: bool = False):
        return not quant

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        return np.allclose(out_data, out_data_ref, rtol=self.rtol, atol=self.atol)


class TopKMetric(ValidationMetric):
    def __init__(self, name: str, n: int = 2):
        super().__init__(name)
        assert n >= 1
        self.n = n

    def check(self, out_data, out_data_ref, quant: bool = False):
        data_len = len(out_data.flatten().tolist())
        # Probably no classification
        return data_len < 25 and not quant

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        # TODO: only for classification models!
        # TODO: support multi_outputs?
        data_sorted_idx = list(reversed(np.argsort(out_data).tolist()[0]))
        ref_data_sorted_idx = list(reversed(np.argsort(out_data_ref).tolist()[0]))
        k = 0
        num_checks = min(self.n, len(data_sorted_idx))
        assert len(data_sorted_idx) == len(ref_data_sorted_idx)
        for j in range(num_checks):
            idx = data_sorted_idx[j]
            ref_idx = ref_data_sorted_idx[j]
            if idx == ref_idx:
                k += 1
            else:
                if out_data.tolist()[0][idx] == out_data_ref.tolist()[0][ref_idx]:
                    k += 1
                else:
                    break
        if k < num_checks:
            return False
        elif k == num_checks:
            return True
        else:
            assert False


class TopKLabelsMetric(ClassifyMetric):
    def __init__(self, name: str, n: int = 2):
        super().__init__(name)
        assert n >= 1
        self.n = n

    def check(self, out_data, label_ref, quant: bool = False):
        data_len = len(out_data.flatten().tolist())
        # Probably no classification
        return data_len < 25

    def process_(self, out_data, label_ref, quant: bool = False):
        # print("process_")
        # print("out_data", out_data)
        # print("label_ref", label_ref)
        data_sorted_idx = list(reversed(np.argsort(out_data).tolist()[0]))
        # print("data_sorted_idx", data_sorted_idx)
        data_sorted_idx_trunc = data_sorted_idx[: self.n]
        # print("data_sorted_idx_trunc", data_sorted_idx_trunc)
        res = label_ref in data_sorted_idx_trunc
        # print("res", res)
        # TODO: handle same values?
        # input("111")
        return res


class ConfusionMatrixMetric(ValidationMetric):
    def __init__(self, name: str):
        super().__init__(name)
        self.temp = {}
        self.num_correct_per_class = {}

    def check(self, out_data, label_ref, quant: bool = False):
        data_len = len(out_data.flatten().tolist())
        # Probably no classification
        return data_len < 25 and not quant

    def process_(self, out_data, label_ref, quant: bool = False):
        data_sorted_idx = list(reversed(np.argsort(out_data).tolist()[0]))
        label = data_sorted_idx[0]
        correct = label_ref == label
        # TODO: handle same values?
        return correct, label

    def process(self, out_data, label_ref, quant: bool = False):
        # print("ConfusionMatrixMetric.process")
        if not self.check(out_data, label_ref, quant=quant):
            return
        self.num_total += 1
        correct, label = self.process_(out_data, label_ref)
        if correct:
            self.num_correct += 1
            if label_ref not in self.num_correct_per_class:
                self.num_correct_per_class[label_ref] = 0
            self.num_correct_per_class[label_ref] += 1
        temp_ = self.temp.get(label_ref, {})
        if label not in temp_:
            temp_[label] = 0
        temp_[label] += 1
        self.temp[label_ref] = temp_

    def get_summary(self):
        if self.num_total == 0:
            return "N/A"
        return f"{self.temp}"


class AccuracyMetric(TopKMetric):
    def __init__(self, name: str):
        super().__init__(name, n=1)


class AccuracyLabelsMetric(TopKLabelsMetric):
    def __init__(self, name: str):
        super().__init__(name, n=1)


class MSEMetric(ValidationMetric):
    def __init__(self, name: str, thr: int = 0.5):
        super().__init__(name)
        assert thr >= 0
        self.thr = thr

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        mse = ((out_data - out_data_ref) ** 2).mean()
        return mse < self.thr


class ToyScoreMetric(ValidationMetric):
    def __init__(self, name: str, atol: float = 0.1, rtol: float = 0.1):
        super().__init__(name)
        assert atol >= 0
        self.atol = atol
        assert rtol >= 0
        self.rtol = rtol

    def check(self, out_data, out_data_ref, quant: bool = False):
        data_len = len(out_data.flatten().tolist())
        return data_len == 640 and not quant

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        assert in_data is not None
        in_data_flat = in_data.flatten().tolist()
        out_data_flat = out_data.flatten().tolist()
        ref_out_data_flat = out_data_ref.flatten().tolist()
        res = 0
        ref_res = 0
        length = len(out_data_flat)
        for jjj in range(length):
            res = in_data_flat[jjj] - out_data_flat[jjj]
            res += res**2
            ref_res = in_data_flat[jjj] - ref_out_data_flat[jjj]
            ref_res += ref_res**2
        res /= length
        ref_res /= length
        # print("res", res)
        # print("ref_res", ref_res)
        return np.allclose([res], [ref_res], atol=self.atol, rtol=self.rtol)


class PlusMinusOneMetric(ValidationMetric):
    def __init__(self, name: str):
        super().__init__(name)

    def check(self, out_data, out_data_ref, quant: bool = False):
        return "int" in out_data.dtype.str

    def process_(self, out_data, out_data_ref, in_data: Optional[np.array] = None, quant: bool = False):
        data_ = out_data.flatten().tolist()
        ref_data_ = out_data_ref.flatten().tolist()

        length = len(data_)
        for jjj in range(length):
            diff = abs(data_[jjj] - ref_data_[jjj])
            # print("diff", diff)
            if diff > 1:
                # print("r FALSE")
                return False
        return True


LOOKUP = {
    "allclose": AllCloseMetric,
    "topk": TopKMetric,
    "acc": AccuracyMetric,
    "toy": ToyScoreMetric,
    "mse": MSEMetric,
    "+-1": PlusMinusOneMetric,
    "pm1": PlusMinusOneMetric,
}

LABELS_LOOKUP = {
    "topk_label": TopKLabelsMetric,
    "acc_label": AccuracyLabelsMetric,
    "confusion_matrix": ConfusionMatrixMetric,
}


def parse_validate_metric_args(inp):
    ret = {}
    for x in inp.split(","):
        x = x.strip()
        assert "=" in x
        key, val = x.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except Exception as e:
            raise e
        ret[key] = val
    return ret


def parse_validate_metric(inp, lookup=LOOKUP):
    if "(" in inp:
        metric_name, inp_ = inp.split("(")
        assert inp_[-1] == ")"
        inp_ = inp_[:-1]
        metric_args = parse_validate_metric_args(inp_)
    else:
        metric_name = inp
        metric_args = {}
    metric_cls = lookup.get(metric_name, None)
    assert metric_cls is not None, f"Validate metric not found: {metric_name}"
    metric = metric_cls(inp, **metric_args)
    return metric


def parse_validate_metrics(inp):
    ret = []
    for metric_str in inp.split(";"):
        metric = parse_validate_metric(metric_str)
        ret.append(metric)
    return ret


def parse_classify_metrics(inp):
    ret = []
    for metric_str in inp.split(";"):
        metric = parse_validate_metric(metric_str, lookup=LABELS_LOOKUP)
        ret.append(metric)
    return ret
