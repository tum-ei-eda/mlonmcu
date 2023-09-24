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

def parse_bench_results(out, allow_missing=False):
    matches = re.compile(r"# (.+): ([0-9.,E-]+)").findall(out, re.DOTALL)
    ret = {}
    for key, value in matches:
        value = ast.literal_eval(value)
        assert isinstance(value, (int, float))
        ret[key] = value
    for mode in ["Setup", "Run", "Total"]:
        cycles = ret.get(f"{mode} Cycles", None)
        instructions = ret.get(f"{mode} Instructions", None)
        if cycles is None or instructions is None:
            continue
        ret[f"{mode} CPI"] = cycles / instructions
    return ret

def add_bench_metrics(out, metrics, allow_missing=False):
    results = parse_bench_results(out, allow_missing=allow_missing)
    for key, value in results.items():
        optional = "Total" not in key
        metrics.add(key, value, optional)
