"""MLonMCU Spike Target definitions"""

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

class SpikeTarget(Target):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = ["vext"]

    DEFAULTS = {
        "timeout_sec": 0,  # disabled
        "vlen": 0,  # vectorization=off
        "isa": "rv32gc",  # rv32gcv?
        "extra_args": "",
    }
    REQUIRED = ["spike.exe", "spike.pk", "riscv_gcc.install_dir"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            "spike", features=features, config=config, context=context
        )

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def isa(self):
        return str(self.config["isa"])

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
    def timeout_sec(self):
        # 0 = off
        return int(self.config["timeout_sec"])

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        spike_args = []

        spike_args.append(f"--isa={self.isa}")
        
        if len(self.extra_args) > 0:
            spike_args.extend(self.extra_args.split(" "))

        if "vext" in [feature.name for feature in self.features]:  # TODO: remove this
            spike_args.append(f"--varch=vlen:{self.vlen},elen:32")
            raise NotImplementedError
        else:
            assert self.vlen == 0

        if self.timeout_sec > 0:
            raise NotImplementedError

        ret = execute(
                self.spike_exe.resolve(),
                *spike_args,
                self.spike_pk.resolve(),
                program,
                *args,
                **kwargs,
            )
        return ret

    def parse_stdout(self, out):
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            raise RuntimeError("unexpected script output (cycles)")
        cycles = int(float(cpu_cycles.group(1)))
        mips = None  # TODO: parse mips?
        return cycles

    def get_metrics(self, elf, directory, verbose=False):
        if verbose:
            out = self.exec(elf, cwd=directory, live=True)
        else:
            out = self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None
            )
        cycles = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Total Cycles", cycles)

        return metrics

    def get_cmake_args(self):
        ret = super().get_cmake_args()
        ret.append(f"-DRISCV_ELF_GCC_PREFIX={self.riscv_prefix}")
        return ret


if __name__ == "__main__":
    cli(target=SpikeTarget)
