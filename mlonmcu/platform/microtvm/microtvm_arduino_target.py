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


class ArduinoMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    DEFAULTS = {
        **Target.DEFAULTS,
        "project_type": None,
        "warning_as_error": False,
        "arduino_board": "?",
        # "arduino_cli_cmd": None,
        "verbose": False,
        "port": -1,
    }
    REQUIRED = Target.REQUIRED | {"arduino.install_dir"}

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = None
        # self.option_names = ["project_type", "warning_as_error", "arduino_board", "verbose", "port"]
        self.option_names = ["project_type", "warning_as_error", "arduino_board", "port"]
        # self.platform = platform
        # self.template = name2template(name)

    @property
    def arduino_install_dir(self):
        return Path(self.config["arduino.install_dir"])

    @property
    def port(self):
        return Path(self.config["port"])

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update({"arduino_cli_cmd": self.arduino_install_dir / "arduino-cli"})
        return ret

    def update_environment(self, env):
        super().update_environment(env)
        if self.port:
            env["ESPTOOL_PORT"] = self.port  # TODO: required?
