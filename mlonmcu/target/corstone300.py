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


class Corstone300Target(Target):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = ["ethosu"]

    DEFAULTS = {
        "timeout_sec": 0,  # disabled
        "ethosu_num_macs": 256,
        "extra_args": "",
    }
    REQUIRED = ["corstone300.exe"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(
            "corstone300", features=features, config=config, context=context
        )

    @property
    def ethosu_num_macs(self):
        return int(self.config["ethosu_num_macs"])

    @property
    def fvp_exe(self):
        return Path(self.config["corstone300.exe"])

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
            f"ethosu.num_macs={self.ethosu_num_macs}",
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            'mps3_board.uart0.out_file="-"',
            "-C",
            "mps3_board.uart0.unbuffered_output=1",
            "-C",
            "mps3_board.uart0.shutdown_on_eot=1",
        ]

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        fvp_args = []
        fvp_args.extend(self.get_default_fvp_args())
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
        return None

    def get_metrics(self, elf, directory, verbose=False):
        if verbose:
            out = self.exec(elf, cwd=directory, live=True)
        else:
            out = self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None
            )
        _ = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Total Cycles", -1)

        return metrics

    def get_cmake_args(self):
        ret = super().get_cmake_args()
        return ret


if __name__ == "__main__":
    cli(target=Corstone300Target)
