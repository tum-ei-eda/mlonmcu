"""MLonMCU RISC-V Target definitions"""

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


class RISCVTarget(Target):
    """Common base class for RISCV-like targets. Please do not use this as a target itself!"""

    FEATURES = []

    DEFAULTS = {
        "timeout_sec": 0,  # disabled
        "extra_args": "",
        "arch": "rv32gc",
        "abi": "ilp32d",
    }
    REQUIRED = ["riscv_gcc.install_dir", "riscv_gcc.name"]

    @property
    def riscv_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def riscv_basename(self):
        return Path(self.config["riscv_gcc.name"])

    @property
    def arch(self):
        return str(self.config["arch"])

    @property
    def abi(self):
        return str(self.config["abi"])

    @property
    def extra_args(self):
        return str(self.config["extra_args"])

    @property
    def timeout_sec(self):
        return int(self.config["timeout_sec"])

    def get_target_system(self):
        return "generic_riscv"

    def get_cmake_args(self):
        ret = super().get_cmake_args()
        ret.append(f"-DRISCV_ELF_GCC_PREFIX={self.riscv_prefix}")
        ret.append(f"-DRISCV_ELF_GCC_BASENAME={self.riscv_basename}")
        ret.append(f"-DRISCV_ARCH={self.arch}")
        ret.append(f"-DRISCV_ABI={self.abi}")
        return ret

    def get_arch(self):
        return "riscv"
