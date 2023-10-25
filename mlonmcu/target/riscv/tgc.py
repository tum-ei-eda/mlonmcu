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

import os
import re
from pathlib import Path
from typing import List

from mlonmcu.logging import get_logger
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics

from .riscv import RISCVTarget

logger = get_logger()

class TGCTarget(RISCVTarget):

    FEATURES = RISCVTarget.FEATURES + []

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
    }
    REQUIRED = RISCVTarget.REQUIRED + ["tgc.exe"]

    def __init__(self, name="tgc", features = None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def tgc_exe(self):
        return Path(self.config["tgc.exe"])
    

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        #assert len(args) == 0, "at the moment no args supported"
        ret = execute(
            self.tgc_exe.resolve(),
            program,
            *args,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out, handle_exit=None):
        cpu_cycles = re.search(r"(\d+) cycles", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
        return cycles

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        total_cycles = self.parse_stdout(out, handle_exit=handle_exit)

        metrics = Metrics()
        metrics.add("Cycles", total_cycles)

        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        
        return ret



if __name__ == "__main__":
    cli(target=TGCTarget)