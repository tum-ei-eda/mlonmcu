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
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def requires_patch(context: MlonMcuContext):
    return context.environment.has_feature("disable_legalize")


def _validate_tvm(context: MlonMcuContext, params=None):
    # user_vars = context.environment.vars
    patch = bool(params.get("patch", False))
    if patch:
        if not requires_patch(context):
            return False
    return context.environment.has_framework("tvm")


def _validate_tvm_clone(context: MlonMcuContext, params=None):
    user_vars = context.environment.vars
    tvm_src_dir = user_vars.get("tvm.src_dir", None)
    if tvm_src_dir:
        return False
    return _validate_tvm(context, params=params)


def _validate_tvm_build(context: MlonMcuContext, params=None):
    user_vars = context.environment.vars
    use_tlcpack = user_vars.get("tvm.use_tlcpack", False)
    debug_build = user_vars.get("tvm.debug_build", False)
    release_build = user_vars.get("tvm.release_build", True)
    patch = bool(params.get("patch", False))
    dbg = bool(params.get("dbg", False))
    # There is not good reason to build without cmsisnn
    # cmsisnn = bool(params.get("cmsisnn", False))
    # if cmsisnn:
    #     if not (context.environment.has_feature("cmsisnnbyoc") or context.environment.has_feature("muriscvnnbyoc")):
    #         return False
    if patch:
        if not requires_patch(context):
            return False
    if use_tlcpack:
        assert not requires_patch(context)
        return False
    if dbg:
        assert not use_tlcpack
        if not debug_build:
            return False
    else:
        if not release_build:
            return False
    tvm_install_dir = user_vars.get("tvm.install_dir", None)
    if tvm_install_dir:
        return False

    return context.environment.has_framework("tvm")


def _validate_tvm_install(context: MlonMcuContext, params=None):
    return False


@Tasks.provides(["tvm.src_dir", "tvm.configs_dir", "tvm.pythonpath"])
@Tasks.optional(["tvm_extensions.src_dir"])
@Tasks.param("patch", [False, True])  # This is just a temporary workaround until the patch is hopefully upstreamed
@Tasks.validate(_validate_tvm_clone)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the TVM repository."""
    if not params:
        params = {}
    patch = params["patch"]
    flags = utils.makeFlags((patch, "patch"))
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.environment.paths["deps"].path / "src" / tvmName
    tvmPythonPath = tvmSrcDir / "python"
    if rebuild or not utils.is_populated(tvmSrcDir):
        tvmRepo = context.environment.repos["tvm"]
        utils.clone_wrapper(tvmRepo, tvmSrcDir, refresh=rebuild)
        if patch:
            extSrcDir = context.cache["tvm_extensions.src_dir"]
            patchFile = extSrcDir / "tvmc_diff.patch"
            utils.apply(tvmSrcDir, patchFile)

    context.cache["tvm.src_dir", flags] = tvmSrcDir
    context.cache["tvm.configs_dir", flags] = tvmSrcDir / "configs"
    context.cache["tvm.pythonpath", flags] = tvmPythonPath


@Tasks.needs(["tvm.src_dir", "llvm.install_dir"])
@Tasks.provides(["tvm.build_dir", "tvm.lib"])
@Tasks.param("dbg", [False, True])
@Tasks.param("cmsisnn", [True])  # There is no good reason to build without cmsisnn
@Tasks.validate(_validate_tvm_build)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tvm(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Build the TVM framework."""
    if not params:
        params = {}
    flags_ = utils.makeFlags((params["dbg"], "dbg"), (params["cmsisnn"], "cmsisnn"))
    flags = utils.makeFlags((params["dbg"], "dbg"))
    dbg = bool(params["dbg"])
    cmsisnn = bool(params["cmsisnn"])
    # FIXME: Try to use TVM dir outside of src dir to allow multiple versions/dbg etc!
    # This should help: TVM_LIBRARY_PATH -> tvm.build_dir
    tvmName = utils.makeDirName("tvm", flags=flags)
    tvmSrcDir = context.lookup("tvm.src_dir", ())  # params["patch"] does not affect the build
    tvmBuildDir = context.environment.paths["deps"].path / "build" / tvmName
    tvmInstallDir = context.environment.paths["deps"].path / "install" / tvmName
    tvmLib = tvmBuildDir / "libtvm.so"
    user_vars = context.environment.vars
    if rebuild or not utils.is_populated(tvmBuildDir) or not tvmLib.is_file():
        ninja = False
        if "tvm.make_tool" in user_vars:
            if user_vars["tvm.make_tool"] == "ninja":
                ninja = True
        utils.mkdirs(tvmBuildDir)
        cfgFileSrc = Path(tvmSrcDir) / "cmake" / "config.cmake"
        cfgFile = tvmBuildDir / "config.cmake"
        llvmConfig = str(Path(context.cache["llvm.install_dir"]) / "bin" / "llvm-config")
        llvmConfigEscaped = str(llvmConfig).replace("/", "\\/")
        utils.copy(cfgFileSrc, cfgFile)
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_MICRO OFF/USE_MICRO ON/g",
            str(cfgFile),
        )
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_MICRO_STANDALONE_RUNTIME OFF/USE_MICRO_STANDALONE_RUNTIME ON/g",
            str(cfgFile),
        )
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_LLVM \(OFF\|ON\)/USE_LLVM " + llvmConfigEscaped + "/g",
            str(cfgFile),
        )
        utils.exec(
            "sed",
            "-i",
            "--",
            r"s/USE_UMA OFF/USE_UMA ON/g",
            str(cfgFile),
        )
        if cmsisnn:
            utils.exec(
                "sed",
                "-i",
                "--",
                r"s/USE_CMSISNN \(OFF\|ON\)/USE_CMSISNN ON/g",
                str(cfgFile),
            )

        utils.cmake(
            tvmSrcDir,
            "-DCMAKE_INSTALL_PREFIX=" + str(tvmInstallDir),
            cwd=tvmBuildDir,
            debug=dbg,
            use_ninja=ninja,
            live=verbose,
        )
        utils.make(cwd=tvmBuildDir, threads=threads, use_ninja=ninja, live=verbose)
    context.cache["tvm.build_dir", flags] = tvmBuildDir
    context.cache["tvm.build_dir", flags_] = tvmBuildDir
    context.cache["tvm.install_dir", flags] = tvmInstallDir
    context.cache["tvm.install_dir", flags_] = tvmInstallDir
    context.cache["tvm.lib", flags] = tvmLib
    context.cache["tvm.lib", flags_] = tvmLib


@Tasks.needs(["tvm.build_dir"])
@Tasks.provides(["tvm.install_dir"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_tvm_install)
@Tasks.register(category=TaskType.FRAMEWORK)
def install_tvm(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Install TVM."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "tvm.install_dir" in user_vars:
        return False
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # tvmName = utils.makeDirName("tvm", flags=flags)
    # tvmSrcDir = context.cache["tvm.src_dir", ()]
    tvmBuildDir = context.cache["tvm.build_dir", flags]
    tvmInstallDir = context.cache["tvm.install_dir", flags]
    if rebuild or not utils.is_populated(tvmInstallDir):
        utils.make("install", cwd=tvmBuildDir, threads=threads, live=verbose)
    context.cache["tvm.install_dir", flags] = tvmInstallDir


def _validate_tvm_extensions(context: MlonMcuContext, params=None):
    return _validate_tvm_build(context, params=params) and requires_patch(context)


@Tasks.provides(["tvm_extensions.src_dir", "tvm_extensions.wrapper"])
@Tasks.validate(_validate_tvm_extensions)
@Tasks.register(category=TaskType.FEATURE)
def clone_tvm_extensions(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the TVM extensions repository."""
    extName = utils.makeDirName("tvm_extensions")
    extSrcDir = context.environment.paths["deps"].path / "src" / extName
    extWrapper = extSrcDir / "tvmc_wrapper.py"
    if rebuild or not utils.is_populated(extSrcDir):
        extRepo = context.environment.repos["tvm_extensions"]
        utils.clone_wrapper(extRepo, extSrcDir, refresh=rebuild)
    context.cache["tvm_extensions.src_dir"] = extSrcDir
    context.cache["tvm_extensions.wrapper"] = extWrapper
