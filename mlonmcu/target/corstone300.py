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
"""MLonMCU Corstone300 Target definitions"""

import os
import re
import csv
from pathlib import Path

# from mlonmcu.context import MlonMcuContext
from mlonmcu.logging import get_logger

logger = get_logger()

from .common import cli, execute
from .target import Target
from .metrics import Metrics
from .elf import get_results


class Corstone300Target(Target):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = ["ethosu"]

    DEFAULTS = {
        "model": "cortex-m55",  # Options: cortex-m4, cortex-m7, cortex-m55 (Frequency is fixed at 25MHz), Warning: FVP is still M55 based!
        "timeout_sec": 0,  # disabled
        "enable_ethosu": False,
        "ethosu_num_macs": 256,
        "extra_args": "",
        "enable_vext": False,
    }
    REQUIRED = [
        "corstone300.exe",
        "cmsisnn.dir",
        "arm_gcc.install_dir",
    ]  # Actually cmsisnn.dir points to the root CMSIS_5 directory

    def __init__(self, name="corstone300", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def enable_ethosu(self):
        return bool(self.config["enable_ethosu"])

    @property
    def ethosu_num_macs(self):
        return int(self.config["ethosu_num_macs"])

    @property
    def fvp_exe(self):
        return Path(self.config["corstone300.exe"])

    @property
    def gcc_prefix(self):
        return str(self.config["arm_gcc.install_dir"])

    @property
    def cmsisnn_dir(self):
        return Path(self.config["cmsisnn.dir"])

    @property
    def extra_args(self):
        return str(self.config["extra_args"])

    @property
    def timeout_sec(self):
        # 0 = off
        return int(self.config["timeout_sec"])

    def get_default_fvp_args(self):
        return [
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            'mps3_board.uart0.out_file="-"',
            "-C",
            "mps3_board.uart0.unbuffered_output=1",
            "-C",
            'mps3_board.uart0.shutdown_tag="EXITTHESIM"',
            "-C",
            "cpu0.CFGDTCMSZ=15",  # ?
            "-C",
            "cpu0.CFGITCMSZ=15",  # ?
        ]

    def get_ethosu_fvp_args(self):
        return [
            "-C",
            f"ethosu.num_macs={self.ethosu_num_macs}",
            "-C",
            f'ethosu.extra_args="--fast"',
        ]

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        fvp_args = []
        fvp_args.extend(self.get_default_fvp_args())
        if self.enable_ethosu:
            fvp_args.extend(self.get_ethosu_fvp_args())
        if self.timeout_sec > 0:
            fvp_args.extend(["--timelimit", str(self.timeout_sec)])
        if len(self.extra_args) > 0:
            fvp_args.extend(self.extra_args.split(" "))

        if "ethosu" in [feature.name for feature in self.features]:  # TODO: remove this
            raise NotImplementedError

        ret = execute(
            self.fvp_exe.resolve(),
            *fvp_args,
            program,
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
        mips = None  # TODO: parse mips?
        return cycles

    def get_metrics(self, elf, directory, verbose=False):
        if verbose:
            out = self.exec(elf, cwd=directory, live=True)
        else:
            out = self.exec(elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None)
        cycles = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Total Cycles", cycles)
        static_mem = get_results(elf)
        rom_ro, rom_code, rom_misc, ram_data, ram_zdata = (
            static_mem["rom_rodata"],
            static_mem["rom_code"],
            static_mem["rom_misc"],
            static_mem["ram_data"],
            static_mem["ram_zdata"],
        )
        rom_total = rom_ro + rom_code + rom_misc
        ram_total = ram_data + ram_zdata
        metrics.add("Total ROM", rom_total)
        metrics.add("Total RAM", ram_total)
        metrics.add("ROM read-only", rom_ro)
        metrics.add("ROM code", rom_code)
        metrics.add("ROM misc", rom_misc)
        metrics.add("RAM data", ram_data)
        metrics.add("RAM zero-init data", ram_zdata)

        return metrics

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["CMSIS_PATH"] = self.cmsisnn_dir
        ret["ARM_COMPILER_PREFIX"] = self.gcc_prefix
        return ret

    def get_arch(self):
        return "arm"


if __name__ == "__main__":
    cli(target=Corstone300Target)
