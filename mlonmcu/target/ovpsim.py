"""MLonMCU OVPSim Target definitions"""

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

class OVPSimTarget(Target):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = ["vext", "gdbserver"]

    DEFAULTS = {
        "timeout_sec": 0,  # disabled
        "vlen": 32,  # vectorization=off
        "enable_vext": False,
        "enable_fpu": True,
        "variant": "RVB32I",
        # "extensions": "MAFDCV",
        "extensions": "MAFDC",  # rv32gc
        "extra_args": "",
    }
    REQUIRED = ["ovpsim.exe", "riscv_gcc.install_dir"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            "ovpsim", features=features, config=config, context=context
        )

    @property
    def ovpsim_exe(self):
        return Path(self.config["ovpsim.exe"])

    @property
    def variant(self):
        return str(self.config["variant"])

    @property
    def extensions(self):
        return str(self.config["extensions"])

    @property
    def riscv_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def extra_args(self):
        return str(self.config["extra_args"])

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def enable_fpu(self):
        return bool(self.config["enable_fpu"])

    @property
    def enable_vext(self):
        return bool(self.config["enable_vext"])

    @property
    def timeout_sec(self):
        # 0 = off
        return int(self.config["timeout_sec"])

# $SCRIPT_DIR/bin/riscvOVPsimPlus \
#     --program $1 \
#     --variant RVB32I \
#     --override riscvOVPsim/cpu/add_Extensions=MAFDCV \
#     --override riscvOVPsim/cpu/unaligned=T \
#     --override riscvOVPsim/cpu/vector_version=1.0-draft-20210130 \
#     --override riscvOVPsim/cpu/VLEN=$2 \
#     --override riscvOVPsim/cpu/ELEN=32 \
#     --override riscvOVPsim/cpu/mstatus_FS=$3 \
#     --override riscvOVPsim/cpu/mstatus_VS=$3
#     # --trace ?
#     # --port 3333 ?
#     # --gdbconsole ?
    def get_default_ovpsim_args(self):
        args =  ["--variant", self.variant, "--override", f"riscvOVPsim/cpu/add_Extensions={self.extensions}", "--override", "riscvOVPsim/cpu/unaligned=T"]
        if "vext" in [feature.name for feature in self.features]:  # TODO: remove this
            raise NotImplementedError
        if self.enable_vext:
            assert "V" in self.extensions
            args.extend(["--override", "riscvOVPsim/cpu/vector_version=1.0-draft-20210130", "--override", f"riscvOVPsim/cpu/VLEN={self.vlen}", "--override", "riscvOVPsim/cpu/ELEN=32"])
            args.extend(["--override", f"riscvOVPsim/cpu/mstatus_VS={int(self.enable_vext)}"])
        if self.enable_fpu:
            assert "F" in self.extensions
        args.extend(["--override", f"riscvOVPsim/cpu/mstatus_FS={int(self.enable_fpu)}"])
        if False:  # ?
            args.append("--trace")
            args.extend(["--port", "3333"])
            args.append("--gdbconsole")
        return args

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        ovpsim_args = []

        ovpsim_args.extend(["--program", str(program)])
        ovpsim_args.extend(self.get_default_ovpsim_args())
        
        if len(self.extra_args) > 0:
            spike_args.extend(self.extra_args.split(" "))

        if self.timeout_sec > 0:
            raise NotImplementedError

        ret = execute(
                self.ovpsim_exe.resolve(),
                *ovpsim_args,
                *args,  # Does this work?
                **kwargs,
            )
        return ret

    def parse_stdout(self, out):
        cpi = 1
        cpu_cycles = re.search(r"  Simulated instructions:(.*)", out)
        if not cpu_cycles:
            raise RuntimeError("unexpected script output (cycles)")
        cycles = int(cpu_cycles.group(1).replace(",", ""))
        mips = None  # TODO: parse mips?
        mips_match = re.search(r"  Simulated MIPS:(.*)", out)
        if mips_match:
            mips_str = float(mips_match.group(1))
            if "run too short for meaningful result" not in mips:
                mips = float(mips_str)
        return cycles, mips

    def get_metrics(self, elf, directory, verbose=False):
        if verbose:
            out = self.exec(elf, cwd=directory, live=True)
        else:
            out = self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None
            )
        cycles, mips = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Total Cycles", cycles)
        metrics.add("MIPS", cycles, optional=True)

        return metrics

    def get_cmake_args(self):
        # ret = super().get_cmake_args()
        ret = [f"-DTARGET_SYSTEM=spike"]  # TODO: rename to generic_riscv
        ret.append(f"-DRISCV_ELF_GCC_PREFIX={self.riscv_prefix}")
        return ret


if __name__ == "__main__":
    cli(target=OVPSimTarget)
