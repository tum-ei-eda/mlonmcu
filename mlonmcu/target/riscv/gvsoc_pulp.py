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
"""MLonMCU GVSOC/Pulp or Pulpissimo Target definitions"""

import os
import re
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget
from .util import update_extensions_pulp
import shutil

logger = get_logger()


class GvsocPulpTarget(RISCVTarget):
    """Target using a Pulpino-like VP running in the GVSOC simulator"""

    FEATURES = RISCVTarget.FEATURES | {"log_instrs", "xpulp"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "abi": "ilp32",
        "extensions": ["i", "m", "c"],  # TODO overwrite extensions elegantly
        "fpu": None,
        "xpulp_version": None,  # None means that xpulp extension is not used
        "model": "pulp",
    }

    REQUIRED = RISCVTarget.PUPL_GCC_TOOLCHAIN_REQUIRED | {
        "gvsoc.exe",
        "pulp_freertos.support_dir",
        "pulp_freertos.config_dir",
        "pulp_freertos.install_dir",
    }

    def __init__(self, name="gvsoc_pulp", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def gvsoc_script(self):
        return Path(self.config["gvsoc.exe"])

    @property
    def gvsoc_folder(self):
        return Path(self.config["gvsoc.exe"]).parent / "gvsoc"

    @property
    def pulp_freertos_support_dir(self):
        return Path(self.config["pulp_freertos.support_dir"])

    @property
    def pulp_freertos_config_dir(self):
        return Path(self.config["pulp_freertos.config_dir"])

    @property
    def pulp_freertos_install_dir(self):
        return Path(self.config["pulp_freertos.install_dir"])

    @property
    def model(self):
        assert self.config["model"] in ["pulp", "pulpissimo"]
        return self.config["model"]

    @property
    def xpulp_version(self):
        value = self.config["xpulp_version"]
        return value

    @property
    def abi(self):
        value = self.config["abi"]
        return value

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions_pulp(exts, xpulp_version=self.xpulp_version)

    def gvsoc_preparation_env(self):
        return {
            "PULP_RISCV_GCC_TOOLCHAIN": str(self.pulp_gcc_prefix),
            "PULP_CURRENT_CONFIG": f"{self.model}@config_file=chips/{self.model}/{self.model}.json",
            "PULP_CONFIGS_PATH": str(self.pulp_freertos_config_dir),
            "PYTHONPATH": str(self.pulp_freertos_install_dir / "python"),
            "INSTALL_DIR": str(self.pulp_freertos_install_dir),
            "ARCHI_DIR": str(self.pulp_freertos_support_dir / "archi" / "include"),
            "SUPPORT_ROOT": str(self.pulp_freertos_support_dir),
        }

    def get_basic_gvsoc_simulating_arg(self, program):
        gvsoc_simulating_arg = []
        gvsoc_simulating_arg.append(f"--dir={program.parent / 'gvsim'}")
        gvsoc_simulating_arg.append(f"--config-file={self.model}@config_file=chips/{self.model}/{self.model}.json")
        gvsoc_simulating_arg.append("--platform=gvsoc")
        gvsoc_simulating_arg.append(f"--binary={program.stem}")
        gvsoc_simulating_arg.append("prepare")
        gvsoc_simulating_arg.append("run")
        return gvsoc_simulating_arg

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute an executable with given arguments"""
        # create a new folder and copy the compiled program into it
        gvsimDir = program.parent / "gvsim"
        if not os.path.exists(gvsimDir):
            os.makedirs(gvsimDir)
        shutil.copyfile(program, gvsimDir / program.stem)

        # prepare simulation by compile gvsoc according to the chosen chip
        gvsoc_compile_args = []
        gvsoc_compile_args.append("build")
        gvsoc_compile_args.append(f"ARCHI_DIR={self.pulp_freertos_support_dir / 'archi' / 'include'}")

        env = os.environ.copy()
        env.update(self.gvsoc_preparation_env())

        gvsoc_compile_retval = execute(
            "make",
            *gvsoc_compile_args,
            env=env,
            cwd=self.gvsoc_folder,
            *args,
            **kwargs,
        )

        # run simulation
        gvsoc_simulating_arg = self.get_basic_gvsoc_simulating_arg(program)
        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            gvsoc_simulating_arg.extend(extra_args)

        env = os.environ.copy()
        env.update({"PULP_RISCV_GCC_TOOLCHAIN": str(self.pulp_gcc_prefix)})
        simulation_retval = execute(
            str(self.gvsoc_script),
            *gvsoc_simulating_arg,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        return gvsoc_compile_retval + simulation_retval

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
        return f"gvsoc_{self.model}"

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        # ret["RISCV_ARCH"] = "rv32imcxpulpv3"
        ret["RISCV_ABI"] = self.abi
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_model": f"{self.model}-{self.arch}"})
        return ret


if __name__ == "__main__":
    cli(target=GvsocPulpTarget)
