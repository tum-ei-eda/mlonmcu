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
import time
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from .riscv_pext_target import RVPTarget
from .riscv_vext_target import RVVTarget
from .riscv_bext_target import RVBTarget
from .util import update_extensions, sort_extensions_canonical, join_extensions

logger = get_logger()


def filter_unsupported_extensions(exts):
    assert isinstance(exts, set)
    REPLACEMENTS = {
        r"zve\d\d[xfd]": "v",
        r"zvl\d+b": None,
        r"zpsfoperand": "p",
        r"zpn": "p",
        r"zbpo": "p",
        # r"p": ["p", "b"],
        # r"p": ["p", "zba", "zbb", "zbc", "zbs"],
    }
    ret = set()
    for ext in exts:
        ignore = False
        for key, value in REPLACEMENTS.items():
            m = re.compile(key).match(ext)
            if m:
                if value:
                    if isinstance(value, list):
                        assert len(value) > 0
                        ret |= set(value)
                    else:
                        ret.add(value)
                ignore = True
        if not ignore:
            ret.add(ext)
    ret = set(ret)

    return ret


class SpikeTarget(RVPTarget, RVVTarget, RVBTarget):
    """Target using the riscv-isa-sim (Spike) RISC-V simulator."""

    FEATURES = RVPTarget.FEATURES | RVVTarget.FEATURES | RVBTarget.FEATURES | {"cachesim", "log_instrs"}

    DEFAULTS = {
        **RVPTarget.DEFAULTS,
        **RVVTarget.DEFAULTS,
        **RVBTarget.DEFAULTS,
        "spikepk_extra_args": [],
    }
    REQUIRED = RVPTarget.REQUIRED | RVVTarget.REQUIRED | RVBTarget.REQUIRED | {"spike.exe", "spike.pk"}

    def __init__(self, name="spike", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def spikepk_extra_args(self):
        return self.config["spikepk_extra_args"]

    @property
    def extensions(self):
        # exts = RVPTarget.extensions(self) + RVVTarget.extensions(self)
        exts = super().extensions
        return update_extensions(
            exts,
        )

    @property
    def isa(self):
        exts = filter_unsupported_extensions(self.extensions)
        exts_str = join_extensions(sort_extensions_canonical(exts, lower=True))
        return f"rv{self.xlen}{exts_str}"

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        spike_args = []
        spikepk_args = []

        spike_args.append(f"--isa={self.isa}")

        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            spike_args.extend(extra_args)

        spikepk_args.append("-s")

        if len(self.spikepk_extra_args) > 0:
            if isinstance(self.spikepk_extra_args, str):
                extra_args = self.spikepk_extra_args.split(" ")
            else:
                extra_args = self.spikepk_extra_args
            spikepk_args.extend(extra_args)  # I rename args to extra_args because otherwise it overwrites *args

        if self.enable_vext:
            assert self.vlen < 8192, "Spike does not support VLEN >= 8192"
            spike_args.append(f"--varch=vlen:{self.vlen},elen:{self.elen}")
        else:
            # assert self.vlen == 0
            pass

        if self.timeout_sec > 0:
            raise NotImplementedError

        ret = execute(
            self.spike_exe.resolve(),
            *spike_args,
            self.spike_pk.resolve(),
            *spikepk_args,
            program,
            *args,
            cwd=cwd,
            **kwargs,
        )
        return ret, []

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        sim_insns = re.search(r"(\d*) cycles", out)
        if sim_insns:
            sim_insns = int(float(sim_insns.group(1)))
            metrics.add("Simulated Instructions", sim_insns, True)

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
            out, artifacts = self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out, artifacts = self.exec(
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

        return metrics, out, artifacts

    def get_platform_defs(self, platform):
        ret = {}
        ret.update(RVPTarget.get_platform_defs(self, platform))
        ret.update(RVVTarget.get_platform_defs(self, platform))
        ret.update(RVBTarget.get_platform_defs(self, platform))
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        if backend in SUPPORTED_TVM_BACKENDS:
            if optimized_layouts:
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
    cli(target=SpikeTarget)
