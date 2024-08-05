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

import multiprocessing

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_vicuna(context: MlonMcuContext, params=None):
    if not context.environment.has_target("vicuna"):
        return False
    return True


@Tasks.provides(["vicuna.src_dir"])
@Tasks.validate(_validate_vicuna)
@Tasks.register(category=TaskType.TARGET)
def clone_vicuna(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the vicuna repository."""
    vicunaName = utils.makeDirName("vicuna")
    vicunaSrcDir = context.environment.paths["deps"].path / "src" / vicunaName
    user_vars = context.environment.vars
    if "vicuna.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(vicunaSrcDir):
        vicunaRepo = context.environment.repos["vicuna"]
        utils.clone_wrapper(vicunaRepo, vicunaSrcDir, refresh=rebuild)
    context.cache["vicuna.src_dir"] = vicunaSrcDir


# REMOVE THIS AND USE VERILATOR TASK
# @Tasks.needs(["vicuna.src_dir"])
# @Tasks.provides(["vicuna.build_dir", "vicuna.install_dir", "vicuna.exe"])
# @Tasks.validate(_validate_vicuna)  # TODO: add validate_vicuna_build
# @Tasks.register(category=TaskType.TARGET)
# def build_vicuna(
#     context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
# ):
#     """Build vicuna verilator testbench."""
#     if not params:
#         params = {}
#     user_vars = context.environment.vars
#     if "vicuna.exe" in user_vars:  # TODO: also check command line flags?
#         return False
#     vicunaName = utils.makeDirName("vicuna")
#     vicunaSrcDir = context.cache["vicuna.src_dir"]
#     vicunaBuildDir = context.environment.paths["deps"].path / "build" / vicunaName
#     vicunaInstallDir = context.environment.paths["deps"].path / "install" / vicunaName
#     vicunaBin = vicunaInstallDir / "vicuna?"
#     if rebuild or not (utils.is_populated(vicunaInstallDir) and vicunaBin.is_file()):
#         utils.mkdirs(vicunaBuildDir)
#         vicunaArgs = []
#         vicunaArgs.append("--prefix=" + str(vicunaInstallDir))
#         utils.execute(
#             str(vicunaSrcDir / "configure"),
#             *vicunaArgs,
#             cwd=vicunaBuildDir,
#             # live=False,
#             live=verbose,
#         )
#         utils.make(cwd=vicunaBuildDir, threads=threads, live=verbose)
#         utils.make(target="install", cwd=vicunaBuildDir, live=verbose)
#     context.cache["vicuna.build_dir"] = vicunaBuildDir
#     context.cache["vicuna.install_dir"] = vicunaInstallDir
#     context.cache["vicuna.exe"] = vicunaBin
