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


def _validate_cmsisnn(context: MlonMcuContext, params=None):
    if not (context.environment.has_feature("cmsisnn") or context.environment.has_feature("cmsisnnbyoc")):
        return False
    mvei = params.get("mvei", False)
    dsp = params.get("dsp", False)
    target_arch = params.get("target_arch", None)
    if target_arch == "arm":
        if dsp and not context.environment.has_feature("arm_dsp"):
            return False
        if mvei and not context.environment.has_feature("arm_mvei"):
            return False
    else:
        if mvei or dsp:
            return False
    return True


def _validate_cmsis(context: MlonMcuContext, params=None):
    return _validate_cmsisnn(context, params=params) or context.environment.has_target("corstone300")


@Tasks.provides(["cmsisnn.dir"])
@Tasks.validate(_validate_cmsis)
@Tasks.register(category=TaskType.MISC)
def clone_cmsis(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """CMSIS repository."""
    cmsisName = utils.makeDirName("cmsis")
    cmsisSrcDir = Path(context.environment.paths["deps"].path) / "src" / cmsisName
    # TODO: allow to skip this if cmsisnn.dir+cmsisnn.lib are provided by the user and corstone is not used
    # -> move those checks to validate?
    if rebuild or not utils.is_populated(cmsisSrcDir):
        cmsisRepo = context.environment.repos["cmsis"]
        utils.clone(cmsisRepo.url, cmsisSrcDir, branch=cmsisRepo.ref, refresh=rebuild)
    context.cache["cmsisnn.dir"] = cmsisSrcDir


@Tasks.needs(["cmsisnn.dir"])
@Tasks.optional(["riscv_gcc.install_dir", "riscv_gcc.name", "arm_gcc.install_dir"])
@Tasks.provides(["cmsisnn.lib"])
@Tasks.param("dbg", [False, True])
# @Tasks.param("target_arch", ["x86", "riscv", "arm"])
@Tasks.param("target_arch", ["x86"])  # Arm/riscv currently broken
@Tasks.param("mvei", [False, True])
@Tasks.param("dsp", [False, True])
@Tasks.validate(_validate_cmsisnn)
@Tasks.register(category=TaskType.OPT)  # TODO: rename to TaskType.FEATURE?
def build_cmsisnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    target_arch = params["target_arch"]
    mvei = params["mvei"]
    dsp = params["dsp"]
    dbg = params["dbg"]
    flags = utils.makeFlags(
        (True, target_arch),
        (mvei, "mvei"),
        (dsp, "dsp"),
        (dbg, "dbg"),
    )
    cmsisnnName = utils.makeDirName("cmsisnn", flags=flags)
    cmsisnnBuildDir = context.environment.paths["deps"].path / "build" / cmsisnnName
    cmsisnnInstallDir = context.environment.paths["deps"].path / "install" / cmsisnnName
    cmsisnnLib = cmsisnnInstallDir / "libcmsis-nn.a"
    cmsisSrcDir = Path(context.cache["cmsisnn.dir"])
    cmsisnnSrcDir = cmsisSrcDir / "CMSIS" / "NN"
    if rebuild or not utils.is_populated(cmsisnnBuildDir) or not cmsisnnLib.is_file():
        utils.mkdirs(cmsisnnBuildDir)
        cmakeArgs = []
        env = os.environ.copy()
        # utils.cmake("-DTF_SRC=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        if params["target_arch"] == "arm":
            toolchainFile = cmsisSrcDir / "CMSIS" / "DSP" / "gcc.cmake"
            armCpu = "cortex-m55"  # TODO: make this variable?
            cmakeArgs.append(f"-DARM_CPU={armCpu}")
            cmakeArgs.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchainFile}")  # Why does this not set CMAKE_C_COMPILER?
            armBinDir = Path(context.cache["arm_gcc.install_dir"]) / "bin"
            cmakeArgs.append("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
            # Warning: this does not work!
            if dsp:
                cmakeArgs.append("-DARM_MATH_DSP=ON")
            if mvei:
                cmakeArgs.append("-DARM_MATH_MVEI=ON")
            old = env["PATH"]
            env["PATH"] = f"{armBinDir}:{old}"
        elif params["target_arch"] == "riscv":
            riscvPrefix = context.cache["riscv_gcc.install_dir"]
            riscvBasename = context.cache["riscv_gcc.name"]
            cmakeArgs.append(f"-DCMAKE_C_COMPILER={riscvPrefix}/bin/{riscvBasename}-gcc")
            # cmakeArgs.append("-DCMAKE_CXX_COMPILER={riscvprefix}/bin/{riscvBasename}-g++")
            # cmakeArgs.append("-DCMAKE_ASM_COMPILER={riscvprefix}/bin/{riscvBasename}-gcc")
            # cmakeArgs.append("-DCMAKE_EXE_LINKER_FLAGS=\"'-march=rv32gc' '-mabi=ilp32d'\"")  # TODO: How about vext?
            # cmakeArgs.append("-E env LDFLAGS=\"-march=rv32gc -mabi=ilp32d\"")
            # cmakeArgs.append("-E env LDFLAGS=\"-march=rv32gc -mabi=ilp32d\"")
            env["LDFLAGS"] = "-march=rv32gc -mabi=ilp32d"
            cmakeArgs.append("-DCMAKE_SYSTEM_NAME=Generic")
            # TODO: how about linker, objcopy, ar?
        elif params["target_arch"] == "x86":
            pass
        else:
            raise ValueError(f"Target architecture '{target_arch}' is not supported")

        utils.cmake(
            *cmakeArgs,
            str(cmsisnnSrcDir),
            debug=dbg,
            cwd=cmsisnnBuildDir,
            live=verbose,
            env=env,
        )
        utils.make(cwd=cmsisnnBuildDir, threads=threads, live=verbose)
        utils.mkdirs(cmsisnnInstallDir)
        utils.move(cmsisnnBuildDir / "Source" / "libcmsis-nn.a", cmsisnnLib)
    context.cache["cmsisnn.lib", flags] = cmsisnnLib
