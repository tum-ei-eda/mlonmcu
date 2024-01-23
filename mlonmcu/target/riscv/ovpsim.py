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
"""MLonMCU OVPSim Target definitions"""

import os
import re
from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target.common import cli, execute
from mlonmcu.target.metrics import Metrics
from .riscv import RISCVTarget, sort_extensions_canonical
from .util import update_extensions

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


class OVPSimTarget(RISCVTarget):
    """Target using an ARM FVP (fixed virtual platform) based on a Cortex M55 with EthosU support"""

    FEATURES = RISCVTarget.FEATURES | {"vext", "pext", "gdbserver", "log_instrs", "trace"}

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "vlen": 0,  # vectorization=off
        "elen": 32,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "enable_pext": False,
        "pext_spec": 0.96,
        "bitmanip_spec": 0.94,
        "variant": None,
        "end_to_end_cycles": True,
        "gdbserver_enable": False,
        "gdbserver_attach": False,
        "gdbserver_port": 2222,
    }
    REQUIRED = RISCVTarget.REQUIRED | {"ovpsim.exe"}

    def __init__(self, name="ovpsim", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def ovpsim_exe(self):
        return Path(self.config["ovpsim.exe"])

    @property
    def variant(self):
        temp = self.config["variant"]
        if temp:
            return temp
        else:
            return f"RVB{self.xlen}I"

    @property
    def extensions(self):
        exts = super().extensions
        assert self.pext_spec <= MAX_P_SPEC, f"P-Extension spec {self.pext_spec} not supported by {self.name} target"
        return update_extensions(
            exts,
            pext=self.enable_pext,
            pext_spec=self.pext_spec,
            vext=self.enable_vext,
            elen=self.elen,
            embedded=self.embedded_vext,
            vlen=self.vlen,
            fpu=self.fpu,
        )

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
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def end_to_end_cycles(self):
        value = self.config["end_to_end_cycles"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gdbserver_enable(self):
        value = self.config["gdbserver_enable"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gdbserver_attach(self):
        value = self.config["gdbserver_attach"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def gdbserver_port(self):
        return int(self.config["gdbserver_port"])

    @property
    def vext_spec(self):
        return float(self.config["vext_spec"])

    @property
    def embedded_vext(self):
        value = self.config["embedded_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def pext_spec(self):
        return float(self.config["pext_spec"])

    def get_default_ovpsim_args(self):
        extensions_before = sort_extensions_canonical(self.extensions, lower=False, unpack=True)
        extensions_after = replace_unsupported(extensions_before)
        extensions_str = "".join(sort_extensions_canonical(extensions_after))
        args = [
            "--variant",
            self.variant,
            "--override",
            f"riscvOVPsim/cpu/add_Extensions={extensions_str}",
            "--override",
            "riscvOVPsim/cpu/unaligned=T",
            "--override",
            "riscvOVPsim/cpu/pk/reportExitErrors=T",
            "--finishonopcode",
            "0",
        ]
        if self.enable_pext:
            args.extend(
                [
                    "--override",
                    f"riscvOVPsim/cpu/dsp_version={self.pext_spec}",
                    "--override",
                    f"riscvOVPsim/cpu/bitmanip_version={self.bitmanip_spec}",
                ]
            )
        if self.enable_vext:
            assert self.has_fpu, "V-Extension requires enabled FPU"
            args.extend(
                [
                    "--override",
                    "riscvOVPsim/cpu/vector_version=1.0-draft-20210130",  # TODO: use vext_spec
                    "--override",
                    f"riscvOVPsim/cpu/VLEN={self.vlen}",
                    "--override",
                    f"riscvOVPsim/cpu/ELEN={self.elen}",
                ]
            )
            args.extend(["--override", f"riscvOVPsim/cpu/mstatus_VS={int(self.enable_vext)}"])
        if self.has_fpu:
            assert "f" in self.extensions or "g" in self.extensions
        args.extend(["--override", f"riscvOVPsim/cpu/mstatus_FS={int(self.has_fpu)}"])
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
            raise NotImplementedError

        ret = execute(
            self.ovpsim_exe.resolve(),
            *ovpsim_args,
            *args,  # Does this work?
            cwd=cwd,
            **kwargs,
        )
        return ret

    def parse_stdout(self, out):
        # cpi = 1
        if self.end_to_end_cycles:
            cpu_cycles = re.search(r".*  Simulated instructions:(.*)", out)
        else:
            cpu_cycles = re.search(r".* Total Cycles: (.*)", out)
        if not cpu_cycles:
            raise RuntimeError("unexpected script output (cycles)")
            cycles = None
        else:
            cycles = int(cpu_cycles.group(1).replace(",", ""))
        mips = None  # TODO: parse mips?
        mips_match = re.search(r".*  Simulated MIPS:(.*)", out)
        if mips_match:
            mips_str = float(mips_match.group(1))
            if "run too short for meaningful result" not in mips:
                mips = float(mips_str)
        return cycles, mips

    def get_metrics(self, elf, directory, *args, handle_exit=None):
        out = ""
        if self.print_outputs:
            out += self.exec(elf, *args, cwd=directory, live=True, handle_exit=handle_exit)
        else:
            out += self.exec(
                elf, *args, cwd=directory, live=False, print_func=lambda *args, **kwargs: None, handle_exit=handle_exit
            )
        cycles, mips = self.parse_stdout(out)

        metrics = Metrics()
        metrics.add("Cycles", cycles)
        if mips:
            metrics.add("MIPS", mips, optional=True)

        return metrics, out, []

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_pext:
            major, minor = str(self.pext_spec).split(".")[:2]
            ret["RISCV_RVP_MAJOR"] = major
            ret["RISCV_RVP_MINOR"] = minor
        if self.enable_vext:
            major, minor = str(self.vext_spec).split(".")[:2]
            ret["RISCV_RVV_MAJOR"] = major
            ret["RISCV_RVV_MINOR"] = minor
            ret["RISCV_RVV_VLEN"] = self.vlen
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
                            "desired_layout": "NHWC:HWOI",
                        }
                    )
        return ret


if __name__ == "__main__":
    cli(target=OVPSimTarget)
