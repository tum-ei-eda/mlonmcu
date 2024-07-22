#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
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
"""Helper functions for benchmarking."""
import re
import ast


def parse_bench_results(out, allow_missing=False, target_name=None):
    lines = out.split("\n")
    for i, line in enumerate(lines):
        if "Program start." in line:
            lines = lines[i:]
            break
    for i, line in enumerate(lines):
        search = "Program finish."
        if target_name and target_name == "ara_rtl":
            search = "gram finish."
        if search in line:
            lines = lines[: i + 1]
            break
    out = "\n".join(lines)
    matches = re.compile(r"# (.+): ([0-9.,E-]+)").findall(out)
    ret = {}
    for key, value in matches:
        value = ast.literal_eval(value)
        assert isinstance(value, (int, float))
        ret[key] = value
    for mode in ["Setup", "Run", "Total"]:
        cycles = ret.get(f"{mode} Cycles", None)
        instructions = ret.get(f"{mode} Instructions", None)

        def calc_cpi(cycles, instructions):
            if cycles is None or instructions is None:
                return None
            if cycles == 0 or instructions == 0:
                return None
            return cycles / instructions

        cpi = calc_cpi(cycles, instructions)
        if cpi is not None:
            ret[f"{mode} CPI"] = cpi

        before = f"{mode} Runtime [us]"
        after = f"{mode} Runtime [s]"
        if before in ret.keys():
            ret[after] = ret[before] / 1e6
            del ret[before]
    return ret


def add_bench_metrics(out, metrics, allow_missing=False, target_name=None):
    results = parse_bench_results(out, allow_missing=allow_missing, target_name=target_name)
    for key, value in results.items():
        optional = "Total" not in key
        metrics.add(key, value, optional)
