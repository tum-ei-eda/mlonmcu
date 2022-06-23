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
"""MLonMCU Spike Target definitions"""

import os
import re
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from .common import cli, execute
from .riscv import RISCVTarget
from .metrics import Metrics

logger = get_logger()


class SpikeTarget(RISCVTarget):
    """Target using the riscv-isa-sim (Spike) RISC-V simulator."""

    FEATURES = ["vext", "pext", "cachesim", "log_instrs"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "enable_vext": False,
        "enable_pext": False,
        "vlen": 0,  # vectorization=off
        "spikepk_extra_args": [],
        "end_to_end_cycles": False,
    }
    REQUIRED = RISCVTarget.REQUIRED + ["spike.exe", "spike.pk"]

    def __init__(self, name="spike", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def enable_vext(self):
        return bool(self.config["enable_vext"])

    @property
    def enable_pext(self):
        return bool(self.config["enable_pext"])

    @property
    def arch(self):
        ret = str(self.config["arch"])
        if self.enable_pext:
            if "p" not in ret[2:]:
                ret += "p"
        if self.enable_vext:
            if "v" not in ret[2:]:
                ret += "v"

        return ret

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def spikepk_extra_args(self):
        return self.config["spikepk_extra_args"]

    @property
    def end_to_end_cycles(self):
        return str2bool(self.config["end_to_end_cycles"])

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        spike_args = []
        spikepk_args = []

        spike_args.append(f"--isa={self.arch}")

        if len(self.extra_args) > 0:
            spike_args.extend(self.extra_args)

        if self.end_to_end_cycles:
            spikepk_args.append("-s")

        if len(self.spikepk_extra_args) > 0:
            spikepk_args.extend(self.spikepk_extra_args.split(" "))

        if self.enable_vext:
            assert self.vlen > 0
            spike_args.append(f"--varch=vlen:{self.vlen},elen:32")
        else:
            assert self.vlen == 0

        if self.timeout_sec > 0:
            raise NotImplementedError

        ret = execute(
            self.spike_exe.resolve(),
            *spike_args,
            self.spike_pk.resolve(),
            *spikepk_args,
            program,
            *args,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out):
        if self.end_to_end_cycles:
            cpu_cycles = re.search(r"(\d*) cycles", out)
        else:
            cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
        # mips = None  # TODO: parse mips?
        return cycles

    def get_metrics(self, elf, directory, handle_exit=None, num=None):
        assert num is None
        out = ""
        if self.print_outputs:
            out += self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        cycles = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Total Cycles", cycles)

        return metrics, out, []

    def get_backend_config(self, backend):
        ret = super().get_backend_config(backend)
        if backend in SUPPORTED_TVM_BACKENDS:
            ret.update({"target_model": "spike-rv32"})
            if self.enable_pext:
                pass  # TODO: change graph layout to use SIMD kernels
        return ret


if __name__ == "__main__":
    cli(target=SpikeTarget)
