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

from mlonmcu.target.target import Target
from mlonmcu.feature.features import SUPPORTED_TVM_BACKENDS

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class EtissMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    FEATURES = Target.FEATURES + ["xcorev"]

    DEFAULTS = {
        **Target.DEFAULTS,
        # "project_type": "?",
        "verbose": False,
        "quiet": True,
        "workspace_size_bytes": None,
        # "warning_as_error": True,
        # "compile_definitions": "",
        # "config_main_stack_size": -1,
        # "riscv_path": "?",
        # "etiss_path": "?",
        # "etissvp_script": "?",
        # "etissvp_script_args": "?",
        # "transport": True,
        "enable_xcorevmac": False,
    }
    REQUIRED = Target.REQUIRED + ["microtvm_etiss.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name", "etissvp.script"]

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
    def enable_xcorevmac(self):
        value = self.config["enable_xcorevmac"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "gcc_prefix": self.riscv_gcc_install_dir,
                "gcc_name": self.riscv_gcc_name,
                "etiss_script": self.etiss_script,
                "etiss_args": "",
                # arch
                # abi
                # ?
            }
        )
        return ret

    def get_backend_config(self, backend):
        attrs = "+m,+a,+f,+d,+c".split(",")
        model = "etiss"
        if self.enable_xcorevmac:
            if "+xcorevmac" not in attrs:
                attrs.append("+xcorevmac")
            model = "etiss-xcorevmac"
        attrs = ",".join(attrs)
        if backend in SUPPORTED_TVM_BACKENDS:
            return {
                "target_device": "riscv_cpu",
                # "target_march": "TODO",
                "target_model": model,
                "target_mtriple": self.riscv_gcc_name,
                # "target_mabi": self.mabi,
                "target_mabi": "ilp32d",
                # "target_mattr": self.mattr,
                "target_mattr": attrs,
                # "target_mcpu": f"generic-rv{self.xlen}",
                "target_mcpu": f"generic-rv32",
            }
        return {}
