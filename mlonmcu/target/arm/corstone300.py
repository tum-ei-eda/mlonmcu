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
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.config import str2bool
from mlonmcu.target import Target
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from .util import resolve_cpu_features

logger = get_logger()


class Corstone300Target(Target):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = {"ethosu", "arm_mvei", "arm_dsp"}

    DEFAULTS = {
        **Target.DEFAULTS,
        "model": "cortex-m55",  # Options: cortex-m{0,0plus,1,3,4,7,23,33,35p,55} (Frequency is fixed at 25MHz)
        # Warning: FVP is still M55 based!
        "timeout_sec": 0,  # disabled
        "enable_ethosu": False,
        "enable_fpu": True,
        "enable_mvei": False,  # unused
        "enable_dsp": False,  # unused
        "ethosu_num_macs": 256,
        "extra_args": "",
    }
    REQUIRED = {
        "corstone300.exe",
        "cmsis.dir",
        "cmsisnn.dir",
        "ethosu_platform.dir",
        "arm_gcc.install_dir",
    }  # Actually cmsisnn.dir points to the root CMSIS_5 directory

    def __init__(self, name="corstone300", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def model(self):
        return self.config["model"]

    @property
    def enable_ethosu(self):
        value = self.config["enable_ethosu"]
        return str2bool(value)

    @property
    def enable_fpu(self):
        value = self.config["enable_fpu"]
        return str2bool(value)

    @property
    def enable_mvei(self):
        value = self.config["enable_mvei"]
        return str2bool(value)

    @property
    def enable_dsp(self):
        value = self.config["enable_dsp"]
        return str2bool(value)

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
    def cmsis_dir(self):
        return Path(self.config["cmsis.dir"])

    @property
    def cmsisnn_dir(self):
        return Path(self.config["cmsisnn.dir"])

    @property
    def ethosu_platform_dir(self):
        return Path(self.config["ethosu_platform.dir"])

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
            "-C",
            f"cpu0.FPU={int(self.enable_fpu)}",
            "-C",
            "cpu0.MVE={}".format(
                2 if self.enable_mvei and self.enable_fpu else (1 if self.enable_mvei and not self.enable_fpu else 0)
            ),
        ]

    def get_ethosu_fvp_args(self):
        return [
            "-C",
            f"ethosu.num_macs={self.ethosu_num_macs}",
            "-C",
            'ethosu.extra_args="--fast"',
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
            cwd=cwd,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out, handle_exit=None):
        exit_match = re.search(r"Application exit code: (.*)\.", out)
        if exit_match:
            exit_code = int(exit_match.group(1))
            if handle_exit is not None:
                exit_code = handle_exit(exit_code)
            if exit_code != 0:
                logger.error("Execution failed - " + out)
                raise RuntimeError(f"unexpected exit code: {exit_code}")
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)

        if not cpu_cycles:
            if exit == 0:
                logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
        # mips = None  # TODO: parse mips?
        return cycles

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        cycles = self.parse_stdout(out, handle_exit=handle_exit)

        metrics = Metrics()
        metrics.add("Cycles", cycles)

        return metrics, out, []

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        ret["CMSIS_DIR"] = self.cmsis_dir
        ret["CMSISNN_DIR"] = self.cmsisnn_dir
        ret["ETHOSU_PLATFORM_DIR"] = self.ethosu_platform_dir
        ret["ARM_COMPILER_PREFIX"] = self.gcc_prefix
        cpu, float_abi, fpu = resolve_cpu_features(
            self.model, enable_fp=self.enable_fpu, enable_dsp=self.enable_dsp, enable_mve=self.enable_mvei
        )
        ret["ARM_CPU"] = cpu
        ret["ARM_FLOAT_ABI"] = float_abi
        ret["ARM_FPU"] = fpu
        return ret

    def get_arch(self):
        return "arm"  # TODO: use proper mapping (v6, v7, v8, v8.1...)

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = {}
        cpu, _, _ = resolve_cpu_features(
            self.model, enable_fp=self.enable_fpu, enable_dsp=self.enable_dsp, enable_mve=self.enable_mvei
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            ret = {
                # "target_march": self.get_arch(),
                "target_mtriple": "arm-none-eabi",
                "target_mcpu": self.model,
                # "target_mattr": "?",
                # "target_mabi": self.abi,
                "target_model": f"{self.name}-{cpu}",
            }
            if optimized_schedules:
                ret.update(
                    {
                        "target_device": "arm_cpu",
                    }
                )

                if optimized_layouts:
                    if self.enable_dsp:
                        ret.update({"desired_layout": "NHWC:HWOI"})
        return ret


if __name__ == "__main__":
    cli(target=Corstone300Target)
