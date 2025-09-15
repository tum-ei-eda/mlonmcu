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
"""MLonMCU OVPSimCOREV Target definitions"""

import os
import re
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.timeout import exec_timeout
from mlonmcu.setup.utils import execute
from mlonmcu.target.common import cli
from mlonmcu.target.metrics import Metrics
from mlonmcu.target.bench import add_bench_metrics
from .riscv import RISCVTarget, sort_extensions_canonical

logger = get_logger()


MAX_P_SPEC = 0.96


def replace_unsupported(exts):
    ret = []
    for ext in exts:
        if "ZVE" in ext or "ZVL" in ext:
            ret.append("V")
        if "zpn" in ext.lower() or "zpsfoperand" in ext.lower():
            pass
        elif ext == "P":
            ret.append(ext)
            ret.append("B")
        else:
            ret.append(ext)
    return ret


class COREVOVPSimTarget(RISCVTarget):
    """TODO"""

    FEATURES = RISCVTarget.FEATURES | {"xcorev", "gdbserver", "log_instrs", "trace"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "variant": None,
        "processor": None,
        "fpu": "none",
        "atomic": False,
        "end_to_end_cycles": False,
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
        "enable_xcorevmac": False,
        "enable_xcorevmem": False,
        "enable_xcorevbi": False,
        "enable_xcorevalu": False,
        "enable_xcorevbitmanip": False,
        "enable_xcorevsimd": False,
        "enable_xcorevhwlp": False,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"corev_ovpsim.exe"}

    def __init__(self, name="corev_ovpsim", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def ovpsim_exe(self):
        return Path(self.config["corev_ovpsim.exe"])

    @property
    def variant(self):
        temp = self.config["variant"]
        if temp:
            return temp
        else:
            return "CV32E40P"

    @property
    def processor(self):
        temp = self.config["processor"]
        if temp:
            return temp
        else:
            return "CVE4P"

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

    @property
    def end_to_end_cycles(self):
        value = self.config["end_to_end_cycles"]
        return str2bool(value)

    @property
    def gdbserver_enable(self):
        value = self.config["gdbserver_enable"]
        return str2bool(value)

    @property
    def gdbserver_attach(self):
        value = self.config["gdbserver_attach"]
        return str2bool(value)

    @property
    def gdbserver_port(self):
        return int(self.config["gdbserver_port"])

    def get_default_ovpsim_args(self):
        extensions_before = sort_extensions_canonical(self.extensions, lower=False, unpack=True)
        extensions_after = replace_unsupported(extensions_before)
        extensions_str = "".join(sort_extensions_canonical(extensions_after))
        args = [
            "--variant",
            self.variant,
            "--processorname",
            self.processor,
            "--override",
            f"riscvOVPsim/cpu/add_Extensions={extensions_str}",
            "--override",
            "riscvOVPsim/cpu/unaligned=T",
            "--override",
            "riscvOVPsim/cpu/pk/reportExitErrors=T",
            "--finishonopcode",
            "0",
            "--override",
            "riscvOVPsim/cpu/extension_CVE4P/mcountinhibit_reset=0",
        ]
        if self.gdbserver_enable:
            # args.append("--trace")
            args.extend(["--port", str(self.gdbserver_port)])
            if self.gdbserver_attach:
                args.append("--gdbconsole")
        return args

    def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
        """Use target to execute a executable with given arguments"""
        ovpsim_args = []

        ovpsim_args.extend(["--program", str(program)])
        ovpsim_args.extend(self.get_default_ovpsim_args())

        if len(self.extra_args) > 0:
            if isinstance(self.extra_args, str):
                extra_args = self.extra_args.split(" ")
            else:
                extra_args = self.extra_args
            ovpsim_args.extend(extra_args)  # I rename args to extra_args because otherwise it overwrites *args

        if self.timeout_sec > 0:
            ret = exec_timeout(
                self.timeout_sec,
                self.ovpsim_exe.resolve(),
                *ovpsim_args,
                *args,
                cwd=cwd,
                **kwargs,
            )
        else:
            ret = execute(
                self.ovpsim_exe.resolve(),
                *ovpsim_args,
                *args,  # Does this work?
                cwd=cwd,
                **kwargs,
            )
        return ret

    def parse_exit(self, out):
        exit_code = super().parse_exit(out)
        if exit_code is None:
            # legacy
            exit_code = None
            exit_match = re.search(r"Error \(RISCV/PK_EXIT\) Non-zero exit code: (.*)", out)
            if exit_match:
                exit_code = int(exit_match.group(1))
        return exit_code

    def parse_stdout(self, out, metrics, exit_code=0):
        add_bench_metrics(out, metrics, exit_code != 0, target_name=self.name)
        if self.end_to_end_cycles:
            sim_insns = re.search(r".*  Simulated instructions:(.*)", out)
            sim_insns = int(float(sim_insns.group(1)))
            metrics.add("Simulated Instructions", sim_insns, True)
        mips = None  # TODO: parse mips?
        mips_match = re.search(r".*  Simulated MIPS:(.*)", out)
        if mips_match:
            mips_str = float(mips_match.group(1))
            if "run too short for meaningful result" not in mips:
                mips = float(mips_str)
        if mips:
            metrics.add("MIPS", mips, optional=True)

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

        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=_handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=_handle_exit
            )
        # TODO: get exit code
        exit_code = 0
        metrics = Metrics()
        self.parse_stdout(out, metrics, exit_code=exit_code)

        return metrics, out, []

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        return ret

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = super().get_backend_config(
            backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
        )
        return ret


if __name__ == "__main__":
    cli(target=COREVOVPSimTarget)
