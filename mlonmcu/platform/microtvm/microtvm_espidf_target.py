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

from mlonmcu.logging import get_logger

from .microtvm_template_target import TemplateMicroTvmPlatformTarget

logger = get_logger()


class EspidfMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    DEFAULTS = {
        **Target.DEFAULTS,
        "project_type": "host_driven",
        "verbose": False,
        "board": None,
        "port": None,
        "baud": None,
    }
    REQUIRED = Target.REQUIRED | {"microtvm_espidf.template", "espidf.src_dir", "espidf.install_dir"}

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.microtvm_espidf_template
        # TODO: interate into TVM build config
        self.option_names = [
            "project_type",
            "verbose",
            "idf_path",
            "idf_tools_path",
            "idf_target",
            "idf_serial_port",
            "idf_serial_baud",
            "warning_as_error",
            "compile_definitions",
            "extra_files_tar",
        ]

    @property
    def microtvm_espidf_template(self):
        return Path(self.config["microtvm_espidf.template"])

    @property
    def esp_idf_src_dir(self):
        return Path(self.config["espidf.src_dir"])

    @property
    def esp_idf_install_dir(self):
        return Path(self.config["espidf.install_dir"])

    @property
    def board(self):
        return self.config["board"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def baud(self):
        return self.config["baud"]

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update({"idf_path": self.esp_idf_src_dir})
        ret.update({"idf_tools_path": self.esp_idf_install_dir})
        assert self.board, "microtvm_espidf.board has do be defined"
        ret.update({"idf_target": self.board})
        if self.port:
            ret.update({"idf_serial_port": self.port})
        if self.baud:
            ret.update({"idf_serial_baud": self.baud})
        return ret
