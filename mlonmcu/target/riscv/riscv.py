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
"""MLonMCU RISC-V Target definitions"""

from pathlib import Path

from mlonmcu.logging import get_logger
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.target import Target
from mlonmcu.config import str2list, str2bool
from .util import sort_extensions_canonical, join_extensions, update_extensions, split_extensions

logger = get_logger()


class RISCVTarget(Target):
    """Common base class for RISCV-like targets. Please do not use this as a target itself!"""

    DEFAULTS = {
        **Target.DEFAULTS,
        # Default: rv32gc
        "xlen": 32,
        "embedded": False,
        "compressed": True,
        "atomic": True,
        "multiply": True,
        "extra_args": "",
        "timeout_sec": 0,  # disabled
        "extensions": [],  # Should only be used for unhandled custom exts
        "fpu": "double",  # allowed: none, single, double
        "arch": None,  # Please use above properties if possible
        "abi": None,  # Please use above properties if possible
        "attr": "",  # Please avoid using this directly
    }

    def reconfigure(self):
        # super().reconfigure()
        self.config.update(
            {
                "final_arch": self.arch,
                "final_abi": self.abi,
            }
        )

    @property
    def xlen(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            assert isinstance(arch, str)
            xlen = int(arch[2:4])
            return xlen
        return int(self.config["xlen"])

    @property
    def embedded(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            exts = split_extensions(arch)
            if "e" in exts:
                return True
        value = self.config["embedded"]
        return str2bool(value)

    @property
    def compressed(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            exts = split_extensions(arch)
            if "c" in exts:
                return True
        value = self.config["compressed"]
        return str2bool(value)

    @property
    def atomic(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            exts = split_extensions(arch)
            if "a" in exts or "g" in exts:
                return True
        value = self.config["atomic"]
        return str2bool(value)

    @property
    def multiply(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            exts = split_extensions(arch)
            if "m" in exts or "g" in exts:
                return True
        value = self.config["multiply"]
        return str2bool(value)

    @property
    def extensions(self):
        arch = self.config.get("arch", None)
        exts = self.config.get("extensions", []).copy()
        if isinstance(exts, str):
            exts = str2list(exts)
        assert isinstance(exts, (list, set))
        exts = set(exts)

        if arch is not None:
            assert isinstance(arch, str)
            exts = split_extensions(arch) | exts
        return update_extensions(
            exts,
            fpu=self.fpu,
            embedded=self.embedded,
            compressed=self.compressed,
            atomic=self.atomic,
            multiply=self.multiply,
            pext=False,
            pext_spec=None,
            vext=False,
            elen=None,
            embedded_vext=False,
            vlen=None,
        )

    @property
    def arch(self):
        temp = self.config["arch"]  # TODO: allow underscores and versions
        if temp:
            return temp
        else:
            exts_str = join_extensions(sort_extensions_canonical(self.extensions, lower=True))
            return f"rv{self.xlen}{exts_str}"

    @property
    def abi(self):
        temp = self.config["abi"]
        if temp:
            return temp
        else:
            if self.xlen == 32:
                temp = "ilp32"
            elif self.xlen == 64:
                temp = "lp64"
            else:
                raise RuntimeError(f"Invalid xlen: {self.xlen}")
            if "d" in self.extensions or "g" in self.extensions:
                temp += "d"
            elif "f" in self.extensions:
                temp += "f"
            return temp

    @property
    def attrs(self):
        attr = self.config.get("attr", None)
        attrs = []
        if attr is not None:
            assert isinstance(attr, str)
            attrs = attr.split(",")
            if len(attrs) == 1 and len(attrs[0]) == 0:
                attrs = []
        if len(attrs) == 0:
            for ext in sort_extensions_canonical(self.extensions, lower=True, unpack=True):
                if ext == "i":
                    continue
                attrs.append(f"+{ext}")
            if self.xlen == 64:
                attrs.append("+64bit")
        return attrs

    @property
    def attr(self):
        attrs = self.attrs
        return ",".join(attrs)

    @property
    def extra_args(self):
        ret = self.config["extra_args"]
        if isinstance(ret, str):
            if len(ret) == 0:
                ret = []
            else:
                ret = [ret]  # TODO: properly split quoted args
        return ret

    @property
    def timeout_sec(self):
        return int(self.config["timeout_sec"])

    @property
    def fpu(self):
        arch = self.config.get("arch", None)
        if arch is not None:
            exts = split_extensions(arch)
            if "d" in exts or "g" in exts:
                return "double"
            elif "f" in exts:
                return "single"
            else:
                return "none"
        value = self.config["fpu"]
        if value is None or not value:
            value = "none"
        assert value in ["none", "single", "double"]
        return value

    @property
    def has_fpu(self):
        return self.fpu != "none"

    def get_target_system(self):
        return "generic_riscv"  # TODO: rename to generic-rv32 for compatibility with LLVM

    @property
    def architecture(self):
        return "riscv"

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        if backend in SUPPORTED_TVM_BACKENDS:
            arch_replace = self.arch.replace("imafd", "g")
            arch_split = arch_replace.split("_")
            arch_remove = ["zicsr", "zifencei"]
            arch_clean = "-".join([a for a in arch_split if a not in arch_remove])
            ret = {
                "target_model": f"{self.name}-{arch_clean}",
            }
            if optimized_schedules:
                ret.update(
                    {
                        "target_device": "riscv_cpu",
                        "target_keys": None,
                    }
                )
            return ret
        return {}
