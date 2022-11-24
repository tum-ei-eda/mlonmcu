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
"""MLonMCU ETISS/Pulpino Target definitions"""

import os
import re
import csv
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget
from .util import update_extensions
import shutil

logger = get_logger()


class GvsocPulpTarget(RISCVTarget):
    """Target using a Pulpino-like VP running in the ETISS simulator"""

    FEATURES = RISCVTarget.FEATURES + ["gdbserver", "etissdbg", "trace", "log_instrs", "pext", "vext"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        # "gdbserver_enable": False,
        # "gdbserver_attach": False,
        # "gdbserver_port": 2222,
        # "debug_etiss": False,
        "trace_memory": False,
        # "plugins": ["PrintInstruction"],
        # "plugins": [],
        # "verbose": False,
        "cpu_arch": None,
        # "rom_start": 0x0,
        # "rom_size": 0x800000,  # 8 MB
        # "ram_start": 0x800000,
        # "ram_size": 0x4000000,  # 64 MB
        # "cycle_time_ps": 31250,  # 32 MHz
        # "enable_vext": False,
        # "vext_spec": 1.0,
        # "embedded_vext": False,
        # "enable_pext": False,
        # "pext_spec": 0.96,
        # "vlen": 0,  # vectorization=off
        # "elen": 32,
        # "jit": None,
        # "end_to_end_cycles": False,
    }
    REQUIRED = RISCVTarget.PUPL_GCC_TOOLCHAIN_REQUIRED + [
        "pulp_gcc.install_dir",
        "pulp_gcc.name",
        "pulp_freertos.support_dir",
        "pulp_gcc.name",
        "gvsoc.exe",
        "pulp_freertos.support_dir",
        "pulp_freertos.config_dir",
        "pulp_freertos.install_dir",
    ]  # "gvsoc.install_dir"

    def __init__(self, name="gvsoc_pulp", features=None, config=None):
        super().__init__(name, features=features, config=config)
        # self.metrics_script = Path(self.etiss_src_dir) / "src" / "bare_etiss_processor" / "get_metrics.py"

    @property
    def pulp_gcc_prefix(self):
        return Path(self.config["pulp_gcc.install_dir"])

    @property
    def pulp_gcc_basename(self):
        return Path(self.config["pulp_gcc.name"])

    @property
    def gvsoc_script(self):
        return Path(self.config["gvsoc.exe"])

    @property
    def pulp_freertos_support_dir(self):
        return Path(self.config["pulp_freertos.support_dir"])

    @property
    def pulp_freertos_config_dir(self):
        return Path(self.config["pulp_freertos.config_dir"])

    @property
    def pulp_freertos_install_dir(self):
        return Path(self.config["pulp_freertos.install_dir"])

    # @property
    # def gdbserver_enable(self):
    #     value = self.config["gdbserver_enable"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def gdbserver_attach(self):
    #     value = self.config["gdbserver_attach"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def gdbserver_port(self):
    #     return int(self.config["gdbserver_port"])

    # @property
    # def debug_etiss(self):
    #     value = self.config["debug_etiss"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def trace_memory(self):
        value = self.config["trace_memory"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def plugins(self):
    #     return self.config["plugins"]

    # @property
    # def verbose(self):
    #     value = self.config["verbose"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def rom_start(self):
    #     value = self.config["rom_start"]
    #     return int(value, 0) if not isinstance(value, int) else value

    # @property
    # def rom_size(self):
    #     value = self.config["rom_size"]
    #     return int(value, 0) if not isinstance(value, int) else value

    # @property
    # def ram_start(self):
    #     value = self.config["ram_start"]
    #     return int(value, 0) if not isinstance(value, int) else value

    # @property
    # def ram_size(self):
    #     value = self.config["ram_size"]
    #     return int(value, 0) if not isinstance(value, int) else value

    # @property
    # def cycle_time_ps(self):
    #     return int(self.config["cycle_time_ps"])

    @property
    def cpu_arch(self):
        if self.config.get("cpu_arch", None):
            return self.config["cpu_arch"]
        # elif self.enable_pext or self.enable_vext:
        #     return "RV32IMACFDPV"
        # else:
        #     return "RV32IMACFD"

    # @property
    # def enable_vext(self):
    #     value = self.config["enable_vext"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def enable_pext(self):
    #     value = self.config["enable_pext"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def vlen(self):
    #     return int(self.config["vlen"])

    # @property
    # def elen(self):
    #     return int(self.config["elen"])

    # @property
    # def jit(self):
    #     return self.config["jit"]

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

    # @property
    # def attr(self):
    #     attrs = super().attr.split(",")
    #     if self.enable_vext and f"+zvl{self.vlen}b" not in attrs:
    #         attrs.append(f"+zvl{self.vlen}b")
    #     return ",".join(attrs)

    # @property
    # def end_to_end_cycles(self):
    #     value = self.config["end_to_end_cycles"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def vext_spec(self):
    #     return float(self.config["vext_spec"])

    # @property
    # def embedded_vext(self):
    #     value = self.config["embedded_vext"]
    #     return str2bool(value) if not isinstance(value, (bool, int)) else value

    # @property
    # def pext_spec(self):
    #     return float(self.config["pext_spec"])

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        gvsimDir = program.parent / "gvsim"
        if not os.path.exists(gvsimDir):
            os.makedirs(gvsimDir)
        shutil.copyfile(program, gvsimDir / program.stem)

        gvsoc_args = []
        gvsoc_args.append(f"--dir={gvsimDir}")

        gvsoc_args.append(f"--config-file=pulp@config_file=chips/pulp/pulp.json")
        gvsoc_args.append(f"--platform=gvsoc")
        gvsoc_args.append(f"--binary={program.stem}")
        gvsoc_args.append(f"prepare")
        gvsoc_args.append(f"run")
        gvsoc_args.append(f"--trace=pe0/insn")
        gvsoc_args.append(f"--trace=pe1/insn")

        # prepare simulation by compile gvsoc according to defined archi
        env = os.environ.copy()
        env.update(
            {
                "PULP_RISCV_GCC_TOOLCHAIN": str(self.pulp_gcc_prefix),
                "PULP_CURRENT_CONFIG": "pulpissimo@config_file=chips/pulp/pulp.json",
                "PULP_CONFIGS_PATH": self.pulp_freertos_config_dir,
                "PYTHONPATH": self.pulp_freertos_install_dir / "python",
                "INSTALL_DIR": self.pulp_freertos_install_dir,
                "ARCHI_DIR": self.pulp_freertos_support_dir / "archi" / "include",
                "SUPPORT_ROOT": self.pulp_freertos_support_dir,
            }
        )
        kwargs["live"] = True
        ret1 = execute(
            str(self.gvsoc_script),
            *gvsoc_args,
            env=env,
            *args,
            **kwargs,
        )

        # run simulation
        env = os.environ.copy()
        env.update({"PULP_RISCV_GCC_TOOLCHAIN": str(self.pulp_gcc_prefix)})
        ret2 = execute(
            str(self.gvsoc_script),
            *gvsoc_args,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        return ret1 + ret2

    def parse_stdout(self, out):
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        cpu_instructions = re.search(r"Total Instructions: (.*)", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            logger.warning("unexpected script output (instructions)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
            instructions = int(float(cpu_instructions.group(1)))
        return cycles, instructions

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
        ret["RISCV_ELF_GCC_PREFIX"] = self.pulp_gcc_prefix
        ret["RISCV_ELF_GCC_BASENAME"] = self.pulp_gcc_basename
        ret["RISCV_ARCH"] = "rv32gc"
        ret["RISCV_ABI"] = "ilp32"
        # ret["ETISS_DIR"] = self.etiss_dir
        # ret["PULPINO_ROM_START"] = self.rom_start
        # ret["PULPINO_ROM_SIZE"] = self.rom_size
        # ret["PULPINO_RAM_START"] = self.ram_start
        # ret["PULPINO_RAM_SIZE"] = self.ram_size
        # if self.enable_pext:
        #     major, minor = str(self.pext_spec).split(".")[:2]
        #     ret["RISCV_RVP_MAJOR"] = major
        #     ret["RISCV_RVP_MINOR"] = minor
        # if self.enable_vext:
        #     major, minor = str(self.vext_spec).split(".")[:2]
        #     ret["RISCV_RVV_MAJOR"] = major
        #     ret["RISCV_RVV_MINOR"] = minor
        #     ret["RISCV_RVV_VLEN"] = self.vlen
        return ret

    def get_backend_config(self, backend):
        ret = super().get_backend_config(backend)
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_model": "etissvp"})
            if self.enable_pext or self.enable_vext:
                ret.update(
                    {
                        # Warning: passing kernel layouts does not work with upstream TVM
                        # TODO: allow passing map?
                        "desired_layout": "NHWC:HWOI",
                    }
                )
        return ret


if __name__ == "__main__":
    cli(target=GvsocPulpTarget)
