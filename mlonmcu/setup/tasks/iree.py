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

import shutil
import multiprocessing

# from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_iree(context: MlonMcuContext, params=None):
    del params
    return context.environment.has_framework("iree")


def _validate_iree_clone(context: MlonMcuContext, params=None):
    user_vars = context.environment.vars
    iree_src_dir = user_vars.get("iree.src_dir", None)
    if iree_src_dir:
        return False
    return _validate_iree(context, params=params)


def _validate_iree_build(context: MlonMcuContext, params=None):
    user_vars = context.environment.vars
    iree_install_dir = user_vars.get("iree.install_dir", None)
    if iree_install_dir:
        return False

    return _validate_iree(context, params=params)


def _validate_iree_install(context: MlonMcuContext, params=None):
    return _validate_iree_build(context, params=params)


def _validate_iree_clean(context: MlonMcuContext, params={}):
    if not _validate_iree(context, params=params):
        return False
    user_vars = context.environment.vars
    keep_build_dir = user_vars.get("iree.keep_build_dir", True)
    return not keep_build_dir


@Tasks.provides(["iree.src_dir"])
@Tasks.validate(_validate_iree_clone)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_iree(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the IREE repository."""
    del verbose, threads
    if not params:
        params = {}
    ireeName = utils.makeDirName("iree")
    ireeSrcDir = context.environment.paths["deps"].path / "src" / ireeName
    if rebuild or not utils.is_populated(ireeSrcDir):
        ireeRepo = context.environment.repos["iree"]
        utils.clone_wrapper(ireeRepo, ireeSrcDir, refresh=rebuild)

    context.cache["iree.src_dir"] = ireeSrcDir
    return True


@Tasks.needs(["iree.src_dir", "cmake.exe"])
@Tasks.provides(["iree.build_dir"])
# @Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_iree_build)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_iree(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Build the IREE framework."""
    if not params:
        params = {}
    # flags_ = utils.makeFlags((params["dbg"], "dbg"), (params["cmsisnn"], "cmsisnn"))
    # flags = utils.makeFlags((params["dbg"], "dbg"))
    # dbg = bool(params["dbg"])
    dbg = False
    # ireeName = utils.makeDirName("iree", flags=flags)
    ireeName = utils.makeDirName("iree")
    ireeSrcDir = context.lookup("iree.src_dir", ())  # params["patch"] does not affect the build
    ireeBuildDir = context.environment.paths["deps"].path / "build" / ireeName
    ireeInstallDir = context.environment.paths["deps"].path / "install" / ireeName
    # ireeLib = ireeBuildDir / "?"
    # user_vars = context.environment.vars
    if rebuild or not utils.is_populated(ireeBuildDir):  # or not ireeLib.is_file():
        ninja = True
        utils.mkdirs(ireeBuildDir)
        iree_cmake_args = [
            "-DCMAKE_INSTALL_PREFIX=" + str(ireeInstallDir),
            "-DIREE_ENABLE_ASSERTIONS=ON",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_ENABLE_CPUINFO=OFF",
            "-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF",
            "-DIREE_INPUT_STABLEHLO=OFF",
            "-DIREE_INPUT_TORCH=OFF",
            "-DIREE_INPUT_TOSA=ON",
            "-DIREE_HAL_DRIVER_VULKAN=OFF",
            "-DIREE_TARGET_BACKEND_METAL_SPIRV=OFF",
            "-DIREE_BUILD_PYTHON_BINDINGS=ON",
        ]
        cmake_exe = context.cache["cmake.exe"]
        utils.cmake(
            ireeSrcDir,
            *iree_cmake_args,
            cwd=ireeBuildDir,
            debug=dbg,
            use_ninja=ninja,
            live=verbose,
            cmake_exe=cmake_exe,
        )
        utils.make(cwd=ireeBuildDir, threads=threads, use_ninja=ninja, live=verbose)
    context.cache["iree.build_dir"] = ireeBuildDir
    context.cache["iree.install_dir"] = ireeInstallDir
    return True


@Tasks.needs(["iree.build_dir"])
@Tasks.provides(["iree.install_dir"])
# @Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_iree_install)
@Tasks.register(category=TaskType.FRAMEWORK)
def install_iree(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Install IREE."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "iree.install_dir" in user_vars:
        return False
    ireeBuildDir = context.cache["iree.build_dir"]
    ireeInstallDir = context.cache["iree.install_dir"]
    if rebuild or not utils.is_populated(ireeInstallDir):
        ninja = True
        utils.make("install", cwd=ireeBuildDir, threads=threads, use_ninja=ninja, live=verbose)
    context.cache["iree.install_dir"] = ireeInstallDir
    return True


@Tasks.needs(["iree.install_dir", "iree.build_dir"])
@Tasks.removes(["iree.build_dir"])  # TODO: implement
@Tasks.validate(_validate_iree_clean)
@Tasks.register(category=TaskType.TARGET)
def clean_iree(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Cleanup IREE build dir."""
    ireeBuildDir = context.cache["iree.build_dir"]
    shutil.rmtree(ireeBuildDir)
    del context.cache["iree.build_dir"]
