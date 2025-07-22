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

from mlonmcu.logging import get_logger
from mlonmcu.target.common import cli
from mlonmcu.setup.utils import execute
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics

from .riscv import RISCVTarget

logger = get_logger()


class TGCTarget(RISCVTarget):

    isa_dict = {"tgc5a": ["e"], "tgc5b": ["i"], "tgc5c": ["i", "m", "c"]}

    FEATURES = RISCVTarget.FEATURES

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "extensions": ["i", "m", "c"],
        "fpu": None,
        "atomic": False,
        "iss_args": None,
        "isa": "tgc5c",
        "backend": "interp",
        # TODO: plugins
    }
    REQUIRED = RISCVTarget.REQUIRED | {"tgc.exe", "tgc_bsp.src_dir"}

    def __init__(self, name="tgc", features=None, config=None):
        super().__init__(name, features=features, config=config)
        # if self.isa != "tgc5c":
        #     self.config["extensions"] = self.isa_dict[self.isa]
        if "e" in self.config["extensions"]:
            self.config["abi"] = "ilp32e"

    @property
    def tgc_exe(self):
        return Path(self.config["tgc.exe"])

    @property
    def tgc_bsp_dir(self):
        return Path(self.config["tgc_bsp.src_dir"])

    @property
    def iss_args(self):
        return self.config["iss_args"]

    @property
    def isa(self):
        return self.config["isa"]

    @property
    def backend(self):
        return self.config["backend"]

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        # assert len(args) == 0, "at the moment no args supported"
        tgc_args = []
        if self.iss_args:
            tgc_args.append(self.iss_args)
        if self.isa != "tgc5c":
            cmd = "--isa=" + self.isa
            tgc_args.append(cmd)
        if self.backend != "interp":
            cmd = "--backend"
            tgc_args.append(cmd)
            tgc_args.append(self.backend)
        ret = execute(
            self.tgc_exe.resolve(),
            program,
            *tgc_args,
            *args,
            **kwargs,
        )
        return ret, []

    def parse_stdout(self, out, metrics, exit_code=0):
        # TODO: filter out log prefix!
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        pattern = r"(\d+) cycles during (\d+)ms resulting in ([\d.]+)MIPS"

        match_ = re.search(pattern, out)

        if match_:
            cycles = int(match_.group(1))
            duration = int(match_.group(2))
            mips = float(match_.group(3))
        else:
            logger.warning("unexcpected script output")
            return
        metrics.add("Simulated Instructions", cycles, True)
        metrics.add("MIPS", mips, optional=True)
        metrics.add("Simulation Time [s]", duration, optional=True)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        if self.print_outputs:
            out, artifacts = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out, artifacts = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)

        return metrics, out, artifacts

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["TGC_BSP_DIR"] = self.tgc_bsp_dir
        return ret


if __name__ == "__main__":
    cli(target=TGCTarget)
