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
import time

# from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.ssh_target import SSHTarget
from mlonmcu.target.bench import add_bench_metrics
from .riscv_vext_target import RVVTarget

logger = get_logger()


class CanMvK230SSHTarget(SSHTarget, RVVTarget):
    """TODO."""

    FEATURES = SSHTarget.FEATURES | RVVTarget.FEATURES

    DEFAULTS = {
        **SSHTarget.DEFAULTS,
        **RVVTarget.DEFAULTS,
        "xlen": 64,
        "vlen": 128,
    }
    REQUIRED = SSHTarget.REQUIRED | RVVTarget.REQUIRED

    def __init__(self, name="canmv_k230_ssh", features=None, config=None):
        super().__init__(name, features=features, config=config)

    def exec(self, program, *args, cwd=os.getcwd(), handle_exit=None, **kwargs):
        """Use target to execute a executable with given arguments"""

        if self.enable_vext:
            assert self.vlen == 128, "CanMV K230 only supports VLEN=128"

        if self.timeout_sec > 0:
            raise NotImplementedError

        output = self.exec_via_ssh(program, *args, **kwargs)
        if handle_exit:
            exit_code = handle_exit(0, out=output)
            assert exit_code == 0
        return output

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            # TODO: before or after?
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        start_time = time.time()
        if self.print_outputs:
            out = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out = self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        # TODO: do something with out?
        end_time = time.time()
        diff = end_time - start_time
        # size instead of readelf?

        # TODO: get exit code
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)

        if metrics.has("Simulated Instructions"):
            sim_insns = metrics.get("Simulated Instructions")
            if diff > 0:
                metrics.add("MIPS", (sim_insns / diff) / 1e6, True)

        return metrics, out, []

    def get_platform_defs(self, platform):
        ret = {}
        ret.update(RVVTarget.get_platform_defs(self, platform))
        return ret


if __name__ == "__main__":
    cli(target=CanMvK230SSHTarget)
