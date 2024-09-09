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


def create_zephyr_platform_target(name, platform, base=Target):
    class ZephyrPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
            "port": None,
            "baud": None,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        @property
        def port(self):
            return self.config["port"]

        @property
        def baud(self):
            return self.config["baud"]

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "Zephyr targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            # Zephyr actually wants a project directory, but we only get the elf now. As a workaround we
            # assume the elf is right in the build directory inside the project directory

            ret = self.platform.run(program, self)
            return ret

        def parse_stdout(self, out):
            # cpu_cycles = re.search(r"Total Cycles: (.*)", out)
            # if not cpu_cycles:
            #     logger.warning("unexpected script output (cycles)")
            #     cycles = None
            # else:
            #     cycles = int(float(cpu_cycles.group(1)))
            cpu_time_us = re.search(r"Total Time: (.*) us", out)
            if not cpu_time_us:
                logger.warning("unexpected script output (time_us)")
                time_us = None
            else:
                time_us = int(float(cpu_time_us.group(1)))
            # return cycles, time_us
            return time_us

        def get_metrics(self, elf, directory, handle_exit=None):
            if self.print_outputs:
                out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                out = self.exec(
                    elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
                )
            # cycles, time_us = self.parse_stdout(out)
            time_us = self.parse_stdout(out)

            metrics = Metrics()
            # metrics.add("Cycles", cycles)
            time_s = time_us / 1e6 if time_us is not None else time_us
            metrics.add("Runtime [s]", time_s)

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

    return ZephyrPlatformTarget
