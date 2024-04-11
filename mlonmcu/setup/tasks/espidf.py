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
"""Definition of tasks used to dynamically install MLonMCU dependencies"""

import os
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_espidf(context: MlonMcuContext, params=None):
    return context.environment.has_platform("espidf")


@Tasks.provides(["espidf.src_dir"])
@Tasks.validate(_validate_espidf)
@Tasks.register(category=TaskType.PLATFORM)
def clone_espidf(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the ESP-IDF repository."""
    espidfName = utils.makeDirName("espidf")
    espidfSrcDir = context.environment.paths["deps"].path / "src" / espidfName
    user_vars = context.environment.vars
    if "espidf.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(espidfSrcDir):
        espidfRepo = context.environment.repos["espidf"]
        utils.clone_wrapper(espidfRepo, espidfSrcDir, refresh=rebuild)
    context.cache["espidf.src_dir"] = espidfSrcDir


@Tasks.needs(["espidf.src_dir"])
@Tasks.provides(["espidf.install_dir"])
@Tasks.validate(_validate_espidf)
@Tasks.register(category=TaskType.PLATFORM)
def install_espidf(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install target support for ESP-IDF toolchain."""
    espidfName = utils.makeDirName("espidf")
    espidfInstallDir = context.environment.paths["deps"].path / "install" / espidfName
    espidfSrcDir = context.cache["espidf.src_dir"]  # TODO: This will fail if the espidf.src_dir is user-supplied
    user_vars = context.environment.vars
    if "espidf.install_dir" in user_vars:  # TODO: also check command line flags?
        return False
    boards = ["all"]
    if "espidf.boards" in user_vars:
        boards = user_vars["espidf.boards"]
    if not isinstance(boards, str):
        assert isinstance(boards, list)
        boards = ",".join(boards)
    if not utils.is_populated(espidfInstallDir) or rebuild:
        # Using idf_tools.py directory instead of ./install.sh because we
        # don't want to use espe-idfs python environment
        espidfInstallScript = Path(espidfSrcDir) / "tools" / "idf_tools.py"
        espidfInstallArgs = ["install", f"--targets={boards}"]
        env = os.environ.copy()
        env["IDF_TOOLS_PATH"] = str(espidfInstallDir)
        utils.python(espidfInstallScript, *espidfInstallArgs, live=verbose, env=env)
    context.cache["espidf.install_dir"] = espidfInstallDir
    context.export_paths.add(espidfInstallDir / "bin")
