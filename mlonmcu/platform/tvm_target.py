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


def create_tvm_target(name, platform, base=Target):
    class TvmTarget(base):

        FEATURES = base.FEATURES + []

        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
        }
        REQUIRED = base.REQUIRED + []

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.device = name2device(name)

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        def exec(self, program, *args, num=1, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "TVM targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            ret = self.platform.run(program, self, num=num)
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

        def get_metrics(self, elf, directory, handle_exit=None, num=1):
            if self.print_outputs:
                out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit, num=num)
            else:
                out = self.exec(
                    elf,
                    cwd=directory,
                    live=False,
                    print_func=lambda *args, **kwargs: None,
                    handle_exit=handle_exit,
                    num=num,
                )
            mean_ms, _, _, _, _ = self.parse_stdout(out)

            metrics = Metrics()
            # metrics.add("Total Cycles", cycles)
            time_s = mean_ms / 1e3 if mean_ms is not None else mean_ms
            metrics.add("Mean Runtime [s]", time_s)

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

    return TvmTarget
