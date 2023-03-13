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
"""MLonMCU ARA Target definitions"""

import os
import re
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget
from .util import update_extensions
import shutil

logger = get_logger()


class AraTarget(RISCVTarget):
    """Target using a Pulpino-like VP running in the GVSOC simulator"""

    FEATURES = RISCVTarget.FEATURES + ["log_instrs", "xpulp"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "xlen": 64,
        "abi": "lp64d",
        "extensions": ["g", "c", "v"],  # TODO overwrite extensions elegantly
    }

    REQUIRED = RISCVTarget.ARA_GCC_TOOLCHAIN_REQUIRED + [
        "ara.apps_dir",
        "ara.hardware_dir"
    ]

    def __init__(self, name="ara", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def ara_apps_dir(self):
        return Path(self.config["ara.apps_dir"])

    @property
    def ara_hardware_dir(self):
        return Path(self.config["ara.hardware_dir"])

    @property
    def Vara_tb_verilator(self):
        return Path(self.config["ara.hardware_dir"]) / "build" / "verilator" / "Vara_tb_verilator"

    @property
    def abi(self):
        value = self.config["abi"]
        return value

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
            # pext=self.enable_pext,
            # pext_spec=self.pext_spec,
            # vext=self.enable_vext,
            # elen=self.elen,
            # embedded=self.embedded_vext,
            # fpu=self.fpu,
            # variant=self.gcc_variant,
        )

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute an executable with given arguments"""
        # run simulation
        ara_verilator_arg = ["-l", f"ram,{program}"]
        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            ara_verilator_arg.extend(extra_args)

        env = os.environ.copy()
        ret = execute(
            str(self.Vara_tb_verilator),
            *ara_verilator_arg,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out):
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))

        cpu_instructions = re.search(r"Total Instructions: (.*)", out)
        if not cpu_instructions:
            logger.warning("unexpected script output (instructions)")
            cpu_instructions = None
        else:
            cpu_instructions = int(float(cpu_instructions.group(1)))
        return cycles, cpu_instructions

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        cycles, instructions = self.parse_stdout(out)
        metrics = Metrics()
        metrics.add("Cycles", cycles)
        metrics.add("Instructions", instructions)
        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        # ret["RISCV_ARCH"] = "rv32imcxpulpv3"
        ret["RISCV_ABI"] = self.abi
        ret["ARA_APPS_DIR"] = self.ara_apps_dir
        return ret

    def get_backend_config(self, backend):
        ret = super().get_backend_config(backend)
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_mabi": self.abi})
        return ret


if __name__ == "__main__":
    cli(target=AraTarget)
