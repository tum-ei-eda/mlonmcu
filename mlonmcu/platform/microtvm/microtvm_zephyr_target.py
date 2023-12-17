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


class ZephyrMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
    DEFAULTS = {
        **Target.DEFAULTS,
        "extra_files_tar": None,
        "project_type": "host_driven",
        "zephyr_board": "",
        # "zephyr_base": "?",
        # "west_cmd": "?",
        "verbose": False,
        "warning_as_error": True,
        "compile_definitions": "",
        # "config_main_stack_size": None,
        "config_main_stack_size": "16384",
        "gdbserver_port": None,
        "nrfjprog_snr": None,
        "openocd_serial": None,
        "port": None,  # Workaround to overwrite esptool detection
    }
    REQUIRED = Target.REQUIRED | {"zephyr.install_dir", "zephyr.sdk_dir"}

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = None
        self.option_names = [
            "extra_files_tar",
            "project_type",
            "zephyr_board",
            # "verbose",
            "warning_as_error",
            "compile_definitions",
            "config_main_stack_size",
            "gdbserver_port",
            "nrfjprog_snr",
            "openocd_serial",
        ]
        # self.platform = platform
        # self.template = name2template(name)

    @property
    def zephyr_install_dir(self):
        return Path(self.config["zephyr.install_dir"])

    @property
    def port(self):
        return self.config["port"]

    @property
    def zephyr_sdk_dir(self):
        return Path(self.config["zephyr.sdk_dir"])

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update({"zephyr_base": self.zephyr_install_dir / "zephyr"})
        return ret

    def update_environment(self, env):
        super().update_environment(env)
        env["ZEPHYR_BASE"] = str(self.zephyr_install_dir / "zephyr")
        env["ZEPHYR_SDK_INSTALL_DIR"] = str(self.zephyr_sdk_dir)
        if self.port:
            env["ESPTOOL_PORT"] = self.port
