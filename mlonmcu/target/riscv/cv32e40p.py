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
"""MLonMCU ETISS/Pulpino Target definitions"""

import os
import re
import time
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.timeout import exec_timeout
from mlonmcu.config import str2bool
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from .riscv import RISCVTarget

logger = get_logger()


class CV32E40PTarget(RISCVTarget):
    """Target for running CV32E40P programs in verilated RTL core."""

    FEATURES = RISCVTarget.FEATURES | {"xcorev"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "enable_xcorevmac": False,
        "enable_xcorevmem": False,
        "enable_xcorevbi": False,
        "enable_xcorevalu": False,
        "enable_xcorevbitmanip": False,
        "enable_xcorevsimd": False,
        "enable_xcorevhwlp": False,
        "fpu": "none",  # FIXED
        "atomic": False,  # FIXED
    }
    REQUIRED = RISCVTarget.REQUIRED | {"cv32e40p.verilator_executable"}

    def __init__(self, name="cv32e40p", features=None, config=None):
        super().__init__(name, features=features, config=config)
        # TODO: make optional or move to mlonmcu pkg

    @property
    def verilator_executable(self):
        return self.config["cv32e40p.verilator_executable"]

    @property
    def enable_xcorevmac(self):
        value = self.config["enable_xcorevmac"]
        return str2bool(value)

    @property
    def enable_xcorevmem(self):
        value = self.config["enable_xcorevmem"]
        return str2bool(value)

    @property
    def enable_xcorevbi(self):
        value = self.config["enable_xcorevbi"]
        return str2bool(value)

    @property
    def enable_xcorevalu(self):
        value = self.config["enable_xcorevalu"]
        return str2bool(value)

    @property
    def enable_xcorevbitmanip(self):
        value = self.config["enable_xcorevbitmanip"]
        return str2bool(value)

    @property
    def enable_xcorevsimd(self):
        value = self.config["enable_xcorevsimd"]
        return str2bool(value)

    @property
    def enable_xcorevhwlp(self):
        value = self.config["enable_xcorevhwlp"]
        return str2bool(value)

    @property
    def extensions(self):
        exts = super().extensions
        required = set()
        if "xcorev" not in exts:
            if self.enable_xcorevmac:
                required.add("xcvmac")
            if self.enable_xcorevmem:
                required.add("xcvmem")
            if self.enable_xcorevbi:
                required.add("xcvbi")
            if self.enable_xcorevalu:
                required.add("xcvalu")
            if self.enable_xcorevbitmanip:
                required.add("xcvbitmanip")
            if self.enable_xcorevsimd:
                required.add("xcvsimd")
            if self.enable_xcorevhwlp:
                required.add("xcvhwlp")
        for ext in required:
            if ext not in exts:
                exts.add(ext)
        return exts

    @property
    def attr(self):
        attrs = super().attr.split(",")
        if self.enable_xcorevmac:
            if "xcorevmac" not in attrs:
                attrs.append("+xcvmac")
        if self.enable_xcorevmem:
            if "xcorevmem" not in attrs:
                attrs.append("+xcvmem")
        if self.enable_xcorevbi:
            if "xcorevbi" not in attrs:
                attrs.append("+xcvbi")
        if self.enable_xcorevalu:
            if "xcorevalu" not in attrs:
                attrs.append("+xcvalu")
        if self.enable_xcorevbitmanip:
            if "xcorevbitmanip" not in attrs:
                attrs.append("+xcvbitmanip")
        if self.enable_xcorevsimd:
            if "xcorevsimd" not in attrs:
                attrs.append("+xcvsimd")
        if self.enable_xcorevhwlp:
            if "xcorevhwlp" not in attrs:
                attrs.append("+xcvhwlp")
        return ",".join(attrs)

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        # Verilator needs hex file instead of elf! Make sure it is created as an artifact!
        program_hex = Path(f"{program}.hex")
        assert program_hex.is_file(), "HEX file for CV32E40P is missing"

        verilator_executable_args = [f"+firmware={program_hex}"]
        if len(self.extra_args) > 0:
            verilator_executable_args.extend(self.extra_args.split(" "))

        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                execute,
                self.verilator_executable,
                *verilator_executable_args,
                *args,  # not really supported
                cwd=cwd,
                **kwargs,
            )
        else:
            ret = execute(
                self.verilator_executable,
                *verilator_executable_args,
                *args,  # not really supported
                cwd=cwd,
                **kwargs,
            )
        return ret

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        if exit_code is None:
            # legacy
            exit_match = re.search(r".* EXIT FAILURE: .*", out)
            if exit_match:
                exit_code = -1  # any none-zero value
            else:
                exit_code = 0
        return exit_code

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        sim_insns = re.search(r"#insns=(\d*)\s", out)
        if sim_insns:
            sim_insns = int(float(sim_insns.group(1)))
            metrics.add("Simulated Instructions", sim_insns, True)

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""

        def _handle_exit(code, out=None):
            assert out is not None
            temp = self.parse_exit(out)
            if temp is None:
                temp = code
            if handle_exit is not None:
                temp = handle_exit(temp, out=out)
            return temp

        metrics_file = os.path.join(directory, "metrics.csv")
        if os.path.exists(metrics_file):
            os.remove(metrics_file)

        # TODO: re-enable sim MIPS
        start_time = time.time()
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        end_time = time.time()
        diff = end_time - start_time
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)
        if metrics.has("Simulated Instructions"):
            sim_insns = metrics.get("Simulated Instructions")
            metrics.add("MIPS", (sim_insns / diff) / 1e6, True)

        return metrics, out, []

    def get_target_system(self):
        return self.name

    def get_platform_defs(self, platform):
        assert platform == "mlif"
        ret = super().get_platform_defs(platform)
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        return ret


if __name__ == "__main__":
    cli(target=CV32E40PTarget)
