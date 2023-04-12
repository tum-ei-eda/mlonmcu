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
import re
import os

from mlonmcu.target.target import Target
from mlonmcu.target.metrics import Metrics

from mlonmcu.logging import get_logger

logger = get_logger()


def name2device(name):
    return name.replace("tvm_", "")


def create_tvm_platform_target(name, platform, base=Target):
    class TvmPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.device = name2device(name)

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        @property
        def repeat(self):
            return None  # This is handled at the platform level

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "TVM targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            ret = self.platform.run(program, self)
            return ret

        def parse_stdout(self, out):
            mean_ms = None
            median_ms = None
            max_ms = None
            min_ms = None
            std_ms = None
            found = False
            for line in out.split("\n"):
                if found:
                    match = re.compile(r"\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)").findall(
                        line
                    )
                    assert len(match) == 1
                    groups = match[0]
                    mean_ms, median_ms, max_ms, min_ms, std_ms = (
                        float(groups[0]),
                        float(groups[1]),
                        float(groups[2]),
                        float(groups[3]),
                        float(groups[4]),
                    )
                    break
                if re.compile(r"\s+mean \(ms\)\s+median \(ms\)\s+max \(ms\)\s+min \(ms\)\s+std \(ms\)").match(line):
                    found = True
            return mean_ms, median_ms, max_ms, min_ms, std_ms

        def get_metrics(self, elf, directory, handle_exit=None):
            if self.print_outputs:
                out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                out = self.exec(
                    elf,
                    cwd=directory,
                    live=False,
                    print_func=lambda *args, **kwargs: None,
                    handle_exit=handle_exit,
                )
            mean_ms, _, max_ms, min_ms, _ = self.parse_stdout(out)

            metrics = Metrics()
            mean_s = mean_ms / 1e3 if mean_ms is not None else mean_ms
            min_s = min_ms / 1e3 if min_ms is not None else min_ms
            max_s = max_ms / 1e3 if max_ms is not None else max_ms
            if (
                self.platform.number == 1
                and self.platform.repeat == 1
                and (not self.platform.total_time or self.platform.aggregate != "none")
            ):
                metrics.add("Runtime [s]", mean_s)
            else:
                if self.platform.total_time:
                    metrics.add("Total Runtime [s]", mean_s * self.platform.number)
                if self.platform.aggregate == "all":
                    metrics.add("Average Runtime [s]", mean_s)
                    metrics.add("Min Runtime [s]", min_s)
                    metrics.add("Max Runtime [s]", max_s)
                elif self.platform.aggregate in ["avg", "mean"]:
                    metrics.add("Average Runtime [s]", mean_s)
                elif self.platform.aggregate == "min":
                    metrics.add("Min Runtime [s]", min_s)
                elif self.platform.aggregate == "max":
                    metrics.add("Max Runtime [s]", max_s)

            if self.platform.profile:
                headers = None
                lines = out.split("\n")
                extracted = []

                def extract_cols(line):
                    x = re.compile(r"(([^\s\[\]]+)(\s\S+)*)(\[.*\])?").findall(line)
                    return [y[0] for y in x]

                for line in lines:
                    if "---" in line:
                        break
                    if headers is None:
                        if "Name" in line:
                            headers = extract_cols(line)
                    else:
                        cols = extract_cols(line)
                        data = {headers[i]: val for i, val in enumerate(cols)}
                        extracted.append(data)
                assert len(extracted) > 0
                metrics = {"default": metrics}
                for item in extracted:
                    metrics_ = Metrics()
                    metrics_.add("Runtime [s]", float(item["Duration (us)"]) / 1e6)
                    metrics[item["Name"]] = metrics_

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

        def update_environment(self, env):
            # TODO: implement in base class?
            pass

    return TvmPlatformTarget
