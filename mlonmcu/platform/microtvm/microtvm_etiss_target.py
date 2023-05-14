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
from mlonmcu.config import str2bool
from mlonmcu.target.target import Target
from mlonmcu.target.riscv.util import sort_extensions_canonical, join_extensions
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class EtissMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    FEATURES = Target.FEATURES | {"xcorev"}

    DEFAULTS = {
        **Target.DEFAULTS,
        # "project_type": "?",
        "verbose": False,
        "quiet": True,
        "workspace_size_bytes": None,
        "xlen": 32,
        "extensions": ["i", "m", "a", "c"],  # TODO overwrite extensions elegantly
        "fpu": "double",  # allowed: none, single, double
        "arch": None,
        "abi": None,
        "attr": "",
        "etiss_extra_args": None,
        "enable_xcorevmac": False,
        "enable_xcorevmem": False,
    }
    REQUIRED = Target.REQUIRED | {"microtvm_etiss.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name", "etissvp.script"}

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_etiss_src_dir / "template_project"
        self.option_names = [
            # "extra_files_tar",
            # "project_type",
            "verbose",
            "quiet",
            "workspace_size_bytes",
            # "warning_as_error",
            # "compile_definitions",
            # "config_main_stack_size",
            # "etissvp_script_args",
            # "transport",
        ]

    @property
    def microtvm_etiss_src_dir(self):
        return Path(self.config["microtvm_etiss.src_dir"])

    @property
    def riscv_gcc_install_dir(self):
        return Path(self.config["riscv_gcc.install_dir"])

    @property
    def riscv_gcc_name(self):
        return self.config["riscv_gcc.name"]

    @property
    def etiss_script(self):
        return Path(self.config["etissvp.script"])

    @property
    def etiss_extra_args(self):
        return self.config["etiss_extra_args"]

    @property
    def enable_xcorevmac(self):
        value = self.config["enable_xcorevmac"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_xcorevmem(self):
        value = self.config["enable_xcorevmem"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "gcc_prefix": self.riscv_gcc_install_dir,
                "gcc_name": self.riscv_gcc_name,
                "etiss_script": self.etiss_script,
                "etiss_args": self.etiss_extra_args,
                "arch": self.arch,
                "abi": self.abi,
            }
        )
        return ret

    @property
    def xlen(self):
        return int(self.config["xlen"])

    @property
    def fpu(self):
        value = self.config["fpu"]
        if value is None or not value:
            value = "none"
        assert value in ["none", "single", "double"]
        return value

    @property
    def extensions(self):
        exts = self.config.get("extensions", []).copy()
        if not isinstance(exts, list):
            exts = exts.split(",")
        required = []
        if "g" not in exts:
            if self.fpu in ["single", "double"]:
                required.append("zicsr")
            if self.fpu == "double":
                required.append("d")
                required.append("f")
                required.append("zifencei")
            if self.fpu == "single":
                required.append("f")
        if "xcorev" not in exts:
            if self.enable_xcorevmac:
                required.append("xcorevmac")
            if self.enable_xcorevmem:
                required.append("xcorevmem")
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
                    "target_model": f"etiss-{arch_clean}",
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

    @property
    def architecture(self):
        return "riscv"
