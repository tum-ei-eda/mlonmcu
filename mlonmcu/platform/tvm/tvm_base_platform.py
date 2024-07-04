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
"""TVM Base Platform"""
import tempfile
from pathlib import Path
from ..platform import Platform
from mlonmcu.setup import utils
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.logging import get_logger

logger = get_logger()


# TODO: abstarct
class TvmBasePlatform(Platform):
    """TVM base platform class."""

    FEATURES = set()

    DEFAULTS = {
        "tvmc_custom_script": None,
        "project_dir": None,
    }

    REQUIRED = {"tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"}

    def __init__(self, name, features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.project_dir = Path(path)
        elif self.config["project_dir"] is not None:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    @property
    def tvm_pythonpath(self):
        return self.config["tvm.pythonpath"]

    @property
    def tvm_build_dir(self):
        return self.config["tvm.build_dir"]

    @property
    def tvm_configs_dir(self):
        return self.config["tvm.configs_dir"]

    def invoke_tvmc(self, command, *args, target=None, live=None, **kwargs):
        if live is None:
            live = self.print_outputs
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if target:
            target.update_environment(env)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=live, env=env, **kwargs)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()
