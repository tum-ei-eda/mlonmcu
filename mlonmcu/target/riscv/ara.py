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

logger = get_logger()


class AraTarget(RISCVTarget):
    """Target using a Pulpino-like VP running in the GVSOC simulator"""

    FEATURES = RISCVTarget.FEATURES + ["log_instrs"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "xlen": 64,
        "abi": "lp64d",
        "extensions": ["g", "c", "v"],  # TODO overwrite extensions elegantly
        "nr_lanes": 4,
        "vlen": 4096,
    }

    REQUIRED = RISCVTarget.ARA_GCC_TOOLCHAIN_REQUIRED + [
        "ara.apps_dir",  # for the bsp package usded in compilation
        "ara.hardware_dir",  # for the rtls
        "ara.bender_path",  # for simulation
        "ara.verilator_install_dir",  # for simulation
        "ara.tb_verilator_build_dir"  # actually just a tmp folder, recommanded to be under hardware/build
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
    def ara_bender_path(self):
        return Path(self.config["ara.bender_path"])

    @property
    def ara_verilator_install_dir(self):
        return Path(self.config["ara.verilator_install_dir"])

    @property
    def ara_tb_verilator_build_dir(self):
        return Path(self.config["ara.tb_verilator_build_dir"])

    @property
    def Vara_tb_verilator(self):
        return Path(self.config["ara.tb_verilator_build_dir"]) / "Vara_tb_verilator"

    @property
    def ara_verilator_exe_path(self):
        return Path(self.config["ara.verilator_install_dir"]) / "bin" / "verilator"

    @property
    def abi(self):
        value = self.config["abi"]
        return value

    @property
    def nr_lanes(self):
        value = self.config["nr_lanes"]
        return value

    @property
    def vlen(self):
        value = self.config["vlen"]
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

        # The following is transferred from https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L165-L167
        # The purpose is to populate the /build/verilator folder
        generate_config_args = []
        generate_config_args.extend(["script", "verilator"])
        generate_config_args.extend(["-t", "rtl", "-t", "ara_test", "-t", "cva6_test", "-t", "verilator"])
        # The following is transferred from https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L105
        # attention! the following NOT included https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L66-L75
        generate_config_args.extend(["--define", f"NR_LANES={self.nr_lanes}"])
        generate_config_args.extend(["--define", f"VLEN={self.vlen}"])
        generate_config_args.extend(["--define", "RVV_ARIANE=1"])
        env = os.environ.copy()
        execute(
            "rm",
            "-rf",
            f"{self.ara_tb_verilator_build_dir}",
        )
        execute(
            "mkdir",
            "-p",
            f"{self.ara_tb_verilator_build_dir}",
        )
        generate_config_ret = execute(
            str(self.ara_bender_path),
            *generate_config_args,
            env=env,
            cwd=self.ara_hardware_dir,
            *args,
            **kwargs,
        )
        # generate_config_ret will be written to the config file
        with open(self.ara_tb_verilator_build_dir / f"bender_script_{self.nr_lanes}nr_lanes_{self.vlen}vlen", "w") as f:
            f.write(generate_config_ret)

        # the following is transferred from https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L169-L203
        verilate_the_design_args = []
        verilate_the_design_args.extend(
            ["-f", f"{self.ara_tb_verilator_build_dir}/bender_script_{self.nr_lanes}nr_lanes_{self.vlen}vlen"]
        )
        verilate_the_design_args.append(f"-GNrLanes={self.nr_lanes}")
        verilate_the_design_args.append("-O3")
        verilate_the_design_args.append("-Wno-BLKANDNBLK")
        verilate_the_design_args.append("-Wno-CASEINCOMPLETE")
        verilate_the_design_args.append("-Wno-CMPCONST")
        verilate_the_design_args.append("-Wno-LATCH")
        verilate_the_design_args.append("-Wno-LITENDIAN")
        verilate_the_design_args.append("-Wno-UNOPTFLAT")
        verilate_the_design_args.append("-Wno-UNPACKED")
        verilate_the_design_args.append("-Wno-UNSIGNED")
        verilate_the_design_args.append("-Wno-WIDTH")
        verilate_the_design_args.append("-Wno-WIDTHCONCAT")
        verilate_the_design_args.extend(["--hierarchical", f"{self.ara_hardware_dir}/tb/verilator/waiver.vlt"])
        verilate_the_design_args.extend(["--Mdir", f"{self.ara_tb_verilator_build_dir}"])
        verilate_the_design_args.append("-Itb/dpi")
        verilate_the_design_args.extend(["--compiler", "clang"])
        verilate_the_design_args.extend(["-CFLAGS", "-DTOPLEVEL_NAME=ara_tb_verilator"])
        verilate_the_design_args.extend(["-CFLAGS", f"-DNR_LANES={self.nr_lanes}"])
        verilate_the_design_args.extend(
            ["-CFLAGS", f"-I{self.ara_hardware_dir}/tb/verilator/lowrisc_dv_verilator_memutil_dpi/cpp"]
        )
        verilate_the_design_args.extend(
            ["-CFLAGS", f"-I{self.ara_hardware_dir}/tb/verilator/lowrisc_dv_verilator_memutil_verilator/cpp"]
        )
        verilate_the_design_args.extend(
            ["-CFLAGS", f"-I{self.ara_hardware_dir}/tb/verilator/lowrisc_dv_verilator_simutil_verilator/cpp"]
        )
        verilate_the_design_args.extend(["-LDFLAGS", "-lelf"])
        verilate_the_design_args.append("--exe")
        for cc_file in (self.ara_hardware_dir / "tb" / "verilator" / "lowrisc_dv_verilator_memutil_dpi" / "cpp").glob(
            "*.cc"
        ):
            verilate_the_design_args.append(str(cc_file))
        for cc_file in (
            self.ara_hardware_dir / "tb" / "verilator" / "lowrisc_dv_verilator_memutil_verilator" / "cpp"
        ).glob("*.cc"):
            verilate_the_design_args.append(str(cc_file))
        for cc_file in (
            self.ara_hardware_dir / "tb" / "verilator" / "lowrisc_dv_verilator_simutil_verilator" / "cpp"
        ).glob("*.cc"):
            verilate_the_design_args.append(str(cc_file))
        verilate_the_design_args.append(f"{self.ara_hardware_dir}/tb/verilator/ara_tb.cpp")
        verilate_the_design_args.append("--cc")
        verilate_the_design_args.extend(["--top-module", "ara_tb_verilator"])
        env = os.environ.copy()
        verilate_the_design_ret = execute(
            self.ara_verilator_exe_path,
            *verilate_the_design_args,
            env=env,
            cwd=self.ara_hardware_dir,
            *args,
            **kwargs,
        )

        env = os.environ.copy()
        env["OBJCACHE"] = ""
        verilate_the_design_compile_ret = execute(
            "make",
            "-j4",
            "-f",
            "Vara_tb_verilator.mk",
            env=env,
            cwd=self.ara_tb_verilator_build_dir,
            *args,
            **kwargs,
        )

        # run simulation
        # to add trace: https://github.com/pulp-platform/ara/blob/main/hardware/Makefile#L201
        ara_verilator_arg = ["-l", f"ram,{program}"]
        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            ara_verilator_arg.extend(extra_args)

        env = os.environ.copy()
        simulation_ret = execute(
            str(self.Vara_tb_verilator),
            *ara_verilator_arg,
            env=env,
            cwd=cwd,
            *args,
            **kwargs,
        )
        return verilate_the_design_ret + verilate_the_design_compile_ret + simulation_ret

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
        ret["XLEN"] = self.xlen
        ret["RISCV_ABI"] = self.abi
        ret["ARA_APPS_DIR"] = self.ara_apps_dir
        ret["MLONMCU_ARA_NR_LANES"] = self.nr_lanes
        ret["MLONMCU_ARA_VLEN"] = self.vlen
        ret["CMAKE_VERBOSE_MAKEFILE"] = "BOOL=OFF"
        return ret

    def get_backend_config(self, backend):
        ret = super().get_backend_config(backend)
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_mabi": self.abi})
        return ret


if __name__ == "__main__":
    cli(target=AraTarget)
