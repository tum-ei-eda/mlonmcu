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
from pathlib import Path

from mlonmcu.utils import filter_none
from mlonmcu.target.target import Target
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS
from mlonmcu.config import str2bool
from mlonmcu.target.riscv.util import sort_extensions_canonical, join_extensions

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class SpikeMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    FEATURES = TemplateMicroTvmPlatformTarget.FEATURES + ["vext", "pext"]

    DEFAULTS = {
        **TemplateMicroTvmPlatformTarget.DEFAULTS,
        "verbose": False,
        "quiet": True,
        "workspace_size_bytes": None,
        "xlen": 32,
        "extensions": ["i", "m", "a", "c"],
        "fpu": "double",  # allowed: none, single, double
        "arch": None,
        "abi": None,
        "attr": "",
        "spike_extra_args": None,
        "pk_extra_args": None,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": False,
        "enable_pext": False,
        "pext_spec": 0.92,  # ?
        "vlen": 0,  # vectorization=off
        "elen": 32,
    }
    REQUIRED = Target.REQUIRED + [
        "spike.exe",
        "spike.pk",
        "riscv_gcc.name",
        "riscv_gcc.install_dir",
        "riscv_gcc.variant",
        "microtvm_spike.src_dir",
    ]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_spike_src_dir / "template_project"
        # TODO: interate into TVM build config
        self.option_names = [
            "verbose",
            "quiet",
            "workspace_size_bytes",
            # TODO
        ]

    @property
    def microtvm_spike_src_dir(self):
        return Path(self.config["microtvm_spike.src_dir"])

    @property
    def spike_exe(self):
        return Path(self.config["spike.exe"])

    @property
    def spike_pk(self):
        return Path(self.config["spike.pk"])

    @property
    def riscv_gcc_name(self):
        return self.config["riscv_gcc.name"]

    @property
    def gcc_variant(self):
        return self.config["riscv_gcc.variant"]

    @property
    def riscv_gcc_install_dir(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def tvm_src_dir(self):
        return Path(self.config["tvm.src_dir"])

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def fpu(self):
        value = self.config["fpu"]
        if value is None or not value:
            value = "none"
        assert value in ["none", "single", "double"]
        return value

    @property
    def has_fpu(self):
        return self.fpu != "none"

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

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

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "gcc_prefix": self.riscv_gcc_install_dir,
                "gcc_name": self.riscv_gcc_name,
                "spike_exe": str(self.spike_exe),
                "spike_pk": str(self.spike_pk),
                "arch": self.arch,
                "abi": self.abi,
                "vlen": self.vlen,
                "elen": self.elen,
            }
        )
        return ret

    @property
    def xlen(self):
        return int(self.config["xlen"])

    @property
    def extensions(self):
        exts = self.config.get("extensions", []).copy()
        if not isinstance(exts, list):
            exts = exts.split(",")
        required = []
        if "g" not in exts:
            if self.fpu in ["single", "double"]:
                pass
                # required.append("zicsr")
            if self.fpu == "double":
                required.append("d")
                required.append("f")
                # required.append("zifencei")
            if self.fpu == "single":
                required.append("f")
        for ext in required:
            if ext not in exts:
                exts.append(ext)
        return exts

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
    def attr(self):
        attrs = str(self.config["attr"]).split(",")
        if len(attrs) == 1 and len(attrs[0]) == 0:
            attrs = []
        for ext in sort_extensions_canonical(self.extensions, lower=True, unpack=True):
            if ext == "i":
                continue
            attrs.append(f"+{ext}")
        attrs = list(set(attrs))
        return ",".join(attrs)

    def get_backend_config(self, backend, optimized_layouts=False, optimized_schedules=False):
        ret = {}
        if backend in SUPPORTED_TVM_BACKENDS:
            arch_replace = self.arch.replace("imafd", "g")
            arch_split = arch_replace.split("_")
            arch_remove = ["zicsr", "zifencei"]
            arch_clean = "-".join([a for a in arch_split if a not in arch_remove])
            ret.update(
                {
                    "target_march": self.arch,
                    "target_mtriple": self.riscv_gcc_name,
                    "target_mabi": self.abi,
                    "target_mattr": self.attr,
                    "target_mcpu": f"generic-rv{self.xlen}",
                    "target_model": f"spike-{arch_clean}",
                }
            )
            if optimized_schedules:
                ret.update(
                    {
                        "target_device": "riscv_cpu",
                        "target_keys": None,
                    }
                )
        return ret

    def add_backend_config(self, backend, config, optimized_layouts=False, optimized_schedules=False):
        new = filter_none(
            self.get_backend_config(
                backend, optimized_layouts=optimized_layouts, optimized_schedules=optimized_schedules
            )
        )

        # only allow overwriting non-none values
        # to support accepting user-vars
        new = {key: value for key, value in new.items() if config.get(key, None) is None}
        config.update(new)
