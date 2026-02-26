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

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

# from .cfu import _validate_cfu_playground

logger = get_logger()
Tasks = get_task_factory()


def _validate_yosys(context: MlonMcuContext, params=None):
    # print("_validate_yosys", _validate_yosys)
    return False
    # return _validate_cfu_playground(context, params=params)


@Tasks.provides(["yosys.src_dir"])
@Tasks.validate(_validate_yosys)
@Tasks.register(category=TaskType.TARGET)
def clone_yosys(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    del verbose, threads
    """Clone the yosys repository."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    flags = utils.makeFlags()
    yosysName = utils.makeDirName("yosys", flags=flags)
    yosysSrcDir = context.environment.paths["deps"].path / "src" / yosysName
    if "yosys.src_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        yosysSrcDir = user_vars["yosys.src_dir"]
    else:
        if rebuild or not utils.is_populated(yosysSrcDir):
            yosysRepo = context.environment.repos["yosys"]
            utils.clone_wrapper(yosysRepo, yosysSrcDir, refresh=rebuild)
    context.cache["yosys.src_dir", flags] = yosysSrcDir


@Tasks.needs(["yosys.src_dir"])
@Tasks.provides(["yosys.build_dir", "yosys.install_dir"])
@Tasks.validate(_validate_yosys)
@Tasks.register(category=TaskType.TARGET)
def build_yosys(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the yosys simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags()
    yosysName = utils.makeDirName("yosys", flags=flags)
    yosysSrcDir = context.environment.paths["deps"].path / "src" / yosysName
    yosysBuildDir = context.environment.paths["deps"].path / "build" / yosysName
    yosysInstallDir = context.environment.paths["deps"].path / "install" / yosysName
    utils.mkdirs(yosysBuildDir)
    user_vars = context.environment.vars
    if "yosys.build_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        yosysBuildDir = user_vars["yosys.build_dir"]
    else:
        if rebuild or not utils.is_populated(yosysBuildDir):
            env = os.environ.copy()
            makefileSrc = yosysSrcDir / "Makefile"
            env["PREFIX"] = yosysInstallDir
            utils.make("-f", makefileSrc, cwd=yosysBuildDir, threads=threads, live=verbose, env=env)
            utils.make("install", cwd=yosysBuildDir, threads=threads, live=verbose, env=env)
    context.cache["yosys.build_dir"] = yosysBuildDir
    context.cache["yosys.install_dir"] = yosysInstallDir
    context.export_paths.add(yosysInstallDir)
