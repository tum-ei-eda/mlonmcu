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

from mlonmcu.logging import get_logger

logger = get_logger()


class ValidationMetric:

    def __init__(self, name, **cfg):
        self.name = name
        self.num_total = 0
        self.num_correct = 0

    def process_(self, out_data, out_data_ref, quant: bool = False):
        raise NotImplementedError

    def check(self, out_data, out_data_ref, quant: bool = False):
        return out_data.dtype == out_data_ref.dtype

    def process(self, out_data, out_data_ref, quant: bool = False):
        if not self.check(out_data, out_data_ref, quant=quant):
            return
        self.num_total += 1
        if self.process_(out_data, out_data_ref):
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

    def process_(self, out_data, out_data_ref, quant: bool = False):
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

    def process_(self, out_data, out_data_ref, quant: bool = False):
        # TODO: only for classification models!
        # TODO: support multi_outputs?
        data_sorted_idx = list(reversed(np.argsort(out_data).tolist()[0]))
        ref_data_sorted_idx = list(reversed(np.argsort(out_data_ref).tolist()[0]))
        k = 0
        num_checks = min(self.n, len(data_sorted_idx))
        assert len(data_sorted_idx) == len(ref_data_sorted_idx)
        # print("data_sorted_idx", data_sorted_idx, type(data_sorted_idx))
        # print("ref_data_sorted_idx", ref_data_sorted_idx, type(ref_data_sorted_idx))
        # print("num_checks", num_checks)
        for j in range(num_checks):
            # print("j", j)
            # print(f"data_sorted_idx[{j}]", data_sorted_idx[j], type(data_sorted_idx[j]))
            idx = data_sorted_idx[j]
            # print("idx", idx)
            ref_idx = ref_data_sorted_idx[j]
            # print("ref_idx", ref_idx)
            if idx == ref_idx:
                # print("IF")
                k += 1
            else:
                # print("ELSE")
                if out_data.tolist()[0][idx] == out_data_ref.tolist()[0][ref_idx]:
                    # print("SAME")
                    k += 1
                else:
                    # print("BREAK")
                    break
        # print("k", k)
        if k < num_checks:
            return False
        elif k == num_checks:
            return True
        else:
            assert False


class AccuracyMetric(TopKMetric):

    def __init__(self, name: str):
        super().__init__(name, n=1)


class MSEMetric(ValidationMetric):

    def __init__(self, name: str, thr: int = 0.5):
        super().__init__(name)
        assert thr >= 0
        self.thr = thr

    def process_(self, out_data, out_data_ref, quant: bool = False):
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

    def process_(self, out_data, out_data_ref, quant: bool = False):
        data_flat = out_data.flatten().tolist()
        ref_data_flat = out_data_ref.flatten().tolist()
        res = 0
        ref_res = 0
        length = len(data_flat)
        for jjj in range(length):
            res += data_flat[jjj] ** 2
            ref_res += ref_data_flat[jjj] ** 2
        res /= length
        ref_res /= length
        print("res", res)
        print("ref_res", ref_res)
        return np.allclose([res], [ref_res], atol=self.atol, rtol=self.rtol)


class PlusMinusOneMetric(ValidationMetric):

    def __init__(self, name: str):
        super().__init__(name)

    def check(self, out_data, out_data_ref, quant: bool = False):
        return "int" in out_data.dtype.str

    def process_(self, out_data, out_data_ref, quant: bool = False):
        data_ = out_data.flatten().tolist()
        ref_data_ = out_data_ref.flatten().tolist()

        length = len(data_)
        for jjj in range(length):
            diff = abs(data_[jjj] - ref_data_[jjj])
            print("diff", diff)
            if diff > 1:
                print("r FALSE")
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


def parse_validate_metric(inp):
    if "(" in inp:
        metric_name, inp_ = inp.split("(")
        assert inp_[-1] == ")"
        inp_ = inp_[:-1]
        metric_args = parse_validate_metric_args(inp_)
    else:
        metric_name = inp
        metric_args = {}
    metric_cls = LOOKUP.get(metric_name, None)
    assert metric_cls is not None, f"Validate metric not found: {metric_name}"
    metric = metric_cls(inp, **metric_args)
    return metric


def parse_validate_metrics(inp):
    ret = []
    for metric_str in inp.split(";"):
        metric = parse_validate_metric(metric_str)
        ret.append(metric)
    return ret
