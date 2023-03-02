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


def _validate_cmake(context: MlonMcuContext, params=None):
    if context.environment.has_platform("mlif"):
        return True
    if context.environment.has_framework("tvm"):
        return True
    if context.environment.has_platform("microtvm"):
        return True
    if context.environment.has_platform("espidf"):
        return True
    if context.environment.has_platform("zephyr"):
        return True
    if context.environment.has_feature("muriscvnn"):
        return True
    if context.environment.has_backend("tflmc"):
        return True
    if context.environment.has_backend("utvmcg"):
        return True
    if context.environment.has_backend("etiss_pulpino"):
        return True
    return False


@Tasks.provides(["cmake.src_dir"])
@Tasks.validate(_validate_cmake)
@Tasks.register(category=TaskType.MISC)
def download_cmake(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Fetch the cmake sources."""
    user_vars = context.environment.vars
    version = user_vars.get("cmake.version", "3.25.2")
    # flags = utils.makeFlags((True, version))
    flags = []
    cmakeName = utils.makeDirName("cmake", flags=flags)
    cmakeSrcDir = context.environment.paths["deps"].path / "src" / cmakeName
    if "cmake.exe" in user_vars:
        return False
    if rebuild or not utils.is_populated(cmakeSrcDir):
        if "cmake.dl_url" in user_vars:
            cmakeUrl = user_vars["boost.dl_url"]
            cmakeUrl, cmakeArchive = cmakeUrl.rsplit("/", 1)
        else:
            cmakeUrl = f"https://github.com/Kitware/CMake/releases/download/v{version}/"
            cmakeArchive = f"cmake-{version}.tar.gz"
            # TODO: windows/macos support?
        utils.download_and_extract(cmakeUrl, cmakeArchive, cmakeSrcDir, progress=verbose)
    context.cache["cmake.src_dir"] = cmakeSrcDir


@Tasks.needs(["cmake.src_dir"])
@Tasks.provides(["cmake.install_dir", "cmake.exe"])
@Tasks.validate(_validate_cmake)
@Tasks.register(category=TaskType.MISC)
def build_cmake(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build cmake tool."""
    flags = []
    cmakeName = utils.makeDirName("cmake", flags=flags)
    cmakeSrcDir = context.cache["cmake.src_dir"]
    cmakeBuildDir = context.environment.paths["deps"].path / "build" / cmakeName
    # boostBuildDir = boostSrcDir
    cmakeInstallDir = context.environment.paths["deps"].path / "install" / cmakeName
    cmakeExe = cmakeInstallDir / "bin" / "cmake"
    user_vars = context.environment.vars
    if "cmake.exe" in user_vars:
        return False
    if rebuild or not utils.is_populated(cmakeInstallDir) or not utils.is_file(cmakeExe):
        bootstrapArgs = [f"--prefix={cmakeInstallDir}", "--", "-DCMAKE_USE_OPENSSL=OFF"]
        # env = os.environ.copy()
        utils.mkdirs(cmakeBuildDir)
        utils.exec_getout(
            str(cmakeSrcDir / "bootstrap"),
            *bootstrapArgs,
            cwd=cmakeBuildDir,
            # env=env,
            live=False,
            print_output=False,
        )
        utils.make("install", cwd=cmakeBuildDir, threads=threads, live=verbose)
    context.cache["cmake.install_dir"] = cmakeBuildDir
    context.cache["cmake.exe"] = cmakeExe
