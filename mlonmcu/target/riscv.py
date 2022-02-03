"""MLonMCU RSCV Target definitions"""

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
    }
    REQUIRED = ["riscv_gcc.install_dir"]

    @property
    def riscv_prefix(self):
        return Path(self.config["riscv_gcc.install_dir"])

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
        return ret
