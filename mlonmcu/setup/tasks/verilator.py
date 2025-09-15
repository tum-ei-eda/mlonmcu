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
from .cv32e40p import _validate_cv32e40p

logger = get_logger()
Tasks = get_task_factory()

########
# ara  #
########


def _validate_verilator(context: MlonMcuContext, params=None):
    return (
        context.environment.has_target("ara")
        or context.environment.has_target("ara_rtl")
        or context.environment.has_target("vicuna")
        or _validate_cv32e40p(context, params=params)
    )


@Tasks.provides(["verilator.src_dir"])
@Tasks.validate(_validate_verilator)
@Tasks.register(category=TaskType.TARGET)
def clone_verilator(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the verilator repository."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    flags = utils.makeFlags()
    verilatorName = utils.makeDirName("verilator", flags=flags)
    verilatorSrcDir = context.environment.paths["deps"].path / "src" / verilatorName
    if "verilator.src_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        verilatorSrcDir = user_vars["verilator.src_dir"]
    else:
        if rebuild or not utils.is_populated(verilatorSrcDir):
            verilatorRepo = context.environment.repos["verilator"]
            utils.clone_wrapper(verilatorRepo, verilatorSrcDir, refresh=rebuild)
    context.cache["verilator.src_dir", flags] = verilatorSrcDir


@Tasks.needs(["verilator.src_dir"])
@Tasks.provides(["verilator.build_dir", "verilator.install_dir"])
@Tasks.validate(_validate_verilator)
@Tasks.register(category=TaskType.TARGET)
def build_verilator(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the verilator simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags()
    verilatorName = utils.makeDirName("verilator", flags=flags)
    verilatorSrcDir = context.environment.paths["deps"].path / "src" / verilatorName
    verilatorBuildDir = context.environment.paths["deps"].path / "build" / verilatorName
    verilatorInstallDir = context.environment.paths["deps"].path / "install" / verilatorName
    utils.mkdirs(verilatorBuildDir)
    user_vars = context.environment.vars
    if "verilator.build_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        verilatorBuildDir = user_vars["verilator.build_dir"]
    else:
        if rebuild or not utils.is_populated(verilatorBuildDir):
            env = os.environ.copy()
            env["VERILATOR_ROOT"] = ""
            utils.execute(
                "autoconf",
                env=env,
                cwd=verilatorSrcDir,
                live=verbose,
            )
            env = os.environ.copy()
            utils.execute(
                str(verilatorSrcDir / "configure"),
                f"--prefix={verilatorInstallDir}",
                env=env,
                cwd=verilatorBuildDir,
                # live=verbose,
                live=False,
            )
            utils.make(cwd=verilatorBuildDir, threads=threads, live=verbose)
            utils.make("install", cwd=verilatorBuildDir, threads=threads, live=verbose)
    context.cache["verilator.build_dir"] = verilatorBuildDir
    context.cache["verilator.install_dir"] = verilatorInstallDir


@Tasks.needs(["verilator.build_dir"])
@Tasks.provides(["verilator.install_dir"])
@Tasks.validate(_validate_verilator)
@Tasks.register(category=TaskType.TARGET)
def install_verilator(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Install the verilator simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags()
    verilatorName = utils.makeDirName("verilator", flags=flags)
    verilatorBuildDir = context.environment.paths["deps"].path / "build" / verilatorName
    verilatorInstallDir = context.environment.paths["deps"].path / "install" / verilatorName
    utils.mkdirs(verilatorBuildDir)
    user_vars = context.environment.vars
    if "verilator.install_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        verilatorInstallDir = user_vars["verilator.install_dir"]
    else:
        if rebuild or not utils.is_populated(verilatorInstallDir):
            env = os.environ.copy()
            utils.execute(
                "make",
                "install",
                env=env,
                cwd=verilatorBuildDir,
                live=verbose,
            )
    context.cache["verilator.install_dir"] = verilatorInstallDir
    context.export_paths.add(verilatorInstallDir)
