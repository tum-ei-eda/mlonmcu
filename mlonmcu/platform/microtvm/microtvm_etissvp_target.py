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


class EtissvpMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    FEATURES = Target.FEATURES + []

    DEFAULTS = {
        **Target.DEFAULTS,
        "extra_files_tar": None,
        "project_type": "?",
        "verbose": False,
        "warning_as_error": True,
        "compile_definitions": "",
        "config_main_stack_size": -1,
        # "riscv_path": "?",
        # "etiss_path": "?",
        # "etissvp_script": "?",
        "etissvp_script_args": "?",
        "transport": True,
    }
    REQUIRED = Target.REQUIRED + ["microtvm_etissvp.src_dir", "riscv_gcc.install_dir", "etiss.install_dir"]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_etissvp.src_dir / "template_project"
        self.option_names = [
            "extra_files_tar",
            "project_type",
            # "verbose",
            "warning_as_error",
            "compile_definitions",
            "config_main_stack_size",
            "etissvp_script_args",
            "transport",
        ]

    @property
    def microtvm_etissvp_src_dir(self):
        return Path(self.config["microtvm_etissvp.src_dir"])

    @property
    def riscv_gcc_install_dir(self):
        return Path(self.config["riscv_gcc_install_dir"])

    @property
    def etiss_install_dir(self):
        return Path(self.config["etiss.install_dir"])

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "riscv_path": self.riscv_gcc_install_dir,
                "etiss_path": self.etiss_install_dir,
                "etissvp_script": self.etiss_install_dir / "bin" / "run_helper.sh",
            }
        )
        return ret

    def get_backend_config(self, backend):
        if backend in SUPPORTED_TVM_BACKENDS:
            return {
                "target_device": "riscv_cpu",
                # "target_march": "TODO",
                # "target_model": "TODO",
                # "target_mtriple": "TODO",
                # "target_mabi": "TODO",
                # "target_mattr": "TODO",
                # "target_mcpu": "TODO",
            }
        return {}
