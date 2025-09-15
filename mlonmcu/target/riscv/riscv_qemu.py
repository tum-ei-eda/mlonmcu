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
"""MLonMCU RISC-V QEMU Target definitions"""

import os

import re

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget
from .util import update_extensions

logger = get_logger()

# TODO: create (Riscv)QemuTarget with variable machine


class RiscvQemuTarget(RISCVTarget):
    """Target using a spike machine in the QEMU simulator"""

    FEATURES = RISCVTarget.FEATURES | {"vext"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "vlen": 0,  # TODO: check allowed range [128, 1024]
        "elen": 32,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"riscv32_qemu.exe"}  # TODO: 64 bit?

    def __init__(self, name="riscv_qemu", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def riscv32_qemu_exe(self):
        return self.config["riscv32_qemu.exe"]

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(exts, vext=self.enable_vext, elen=self.elen, embedded=self.embedded_vext, fpu=self.fpu)

    @property
    def attr(self):
        attrs = super().attr.split(",")
        if self.enable_vext and f"+zvl{self.vlen}b" not in attrs:
            attrs.append(f"+zvl{self.vlen}b")
        return ",".join(attrs)

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value)

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value)

    def get_cpu_str(self):
        cfg = {}
        if self.enable_vext:
            cfg["v"] = "true"
            cfg["vlen"] = str(self.vlen)
            cfg["elen"] = str(self.elen)
            cfg["vext_spec"] = "v1.0"
        return ",".join([f"rv{self.xlen}"] + [f"{key}={value}" for key, value in cfg.items()])

    def get_qemu_args(self, program):
        args = []
        args.extend(["-machine", "spike"])
        args.extend(["-bios", "none"])
        args.extend(["-icount", "shift=1"])
        args.extend(["-cpu", self.get_cpu_str()])
        args.append("-nographic")
        args.extend(["-kernel", program])
        return args

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        assert len(args) == 0, "Qemu does not support passing arguments."
        qemu_args = self.get_qemu_args(program)

        if self.timeout_sec > 0:
            raise NotImplementedError
        else:
            ret = execute(
                self.riscv32_qemu_exe,
                *qemu_args,
                cwd=cwd,
                **kwargs,
            )
        return ret

    def parse_stdout(self, out, handle_exit=None):
        cpu_cycles = re.search(r"Total Cycles: (.*)", out)
        if not cpu_cycles:
            logger.warning("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(float(cpu_cycles.group(1)))
        return cycles

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        total_cycles = self.parse_stdout(out, handle_exit=handle_exit)

        metrics = Metrics()
        metrics.add("Cycles", total_cycles)

        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_vext:
            major, minor = str(self.vext_spec).split(".")[:2]
            ret["RISCV_RVV_MAJOR"] = major
            ret["RISCV_RVV_MINOR"] = minor
            ret["RISCV_RVV_VLEN"] = self.vlen
        return ret


if __name__ == "__main__":
    cli(target=RiscvQemuTarget)
