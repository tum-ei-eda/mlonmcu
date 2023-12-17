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

from .tf import _validate_tensorflow
from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_tflite_micro_compiler(context: MlonMcuContext, params=None):
    if not _validate_tensorflow(context, params=params):
        return False
    if not context.environment.has_backend("tflmc"):
        return False
    return True


@Tasks.provides(["tflmc.src_dir"])
@Tasks.validate(_validate_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def clone_tflite_micro_compiler(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the preinterpreter repository."""
    tflmcName = utils.makeDirName("tflmc")
    tflmcSrcDir = context.environment.paths["deps"].path / "src" / tflmcName
    if rebuild or not utils.is_populated(tflmcSrcDir):
        tflmcRepo = context.environment.repos["tflite_micro_compiler"]
        utils.clone_wrapper(tflmcRepo, tflmcSrcDir, refresh=rebuild)
    context.cache["tflmc.src_dir"] = tflmcSrcDir


def _validate_build_tflite_micro_compiler(context: MlonMcuContext, params=None):
    if params:
        muriscvnn = params.get("muriscvnn", False)
        cmsisnn = params.get("cmsisnn", False)
        if muriscvnn and cmsisnn:
            # Not allowed
            return False
        elif muriscvnn:
            if not context.environment.supports_feature("muriscvnn"):
                return False
        elif cmsisnn:
            if not context.environment.supports_feature("cmsisnn"):
                return False
    return _validate_tflite_micro_compiler(context, params=params)


@Tasks.needs(["tflmc.src_dir", "tf.src_dir"])
@Tasks.optional(["muriscvnn.lib", "muriscvnn.inc_dir", "cmsisnn.dir"])
@Tasks.provides(["tflmc.build_dir", "tflmc.exe"])
@Tasks.param("muriscvnn", [False])
@Tasks.param("cmsisnn", [False, True])
@Tasks.param("dbg", [False, True])
@Tasks.param("arch", ["x86"])  # TODO: compile for arm/riscv in the future
@Tasks.validate(_validate_build_tflite_micro_compiler)
@Tasks.register(category=TaskType.BACKEND)
def build_tflite_micro_compiler(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the TFLM preinterpreter."""
    muriscvnn = params.get("muriscvnn", False)
    cmsisnn = params.get("cmsisnn", False)
    dbg = params.get("dbg", False)
    arch = params.get("arch", "x86")
    flags = utils.makeFlags((True, arch), (muriscvnn, "muriscvnn"), (cmsisnn, "cmsisnn"), (dbg, "dbg"))
    # flags_ = utils.makeFlags((dbg, "dbg"))
    tflmcName = utils.makeDirName("tflmc", flags=flags)
    tflmcBuildDir = context.environment.paths["deps"].path / "build" / tflmcName
    tflmcInstallDir = context.environment.paths["deps"].path / "install" / tflmcName
    tflmcExe = tflmcInstallDir / "compiler"
    tfSrcDir = context.cache["tf.src_dir"]
    tflmcSrcDir = context.cache["tflmc.src_dir"]
    if rebuild or not utils.is_populated(tflmcBuildDir) or not tflmcExe.is_file():
        cmakeArgs = [
            "-DTF_SRC=" + str(tfSrcDir),
            "-DGET_TF_SRC=ON",
        ]
        if muriscvnn:
            flags__ = utils.makeFlags((True, arch), (True, "gcc"), (dbg, "dbg"))
            muriscvnnLib = context.cache["muriscvnn.lib", flags__]
            muriscvnnInc = context.cache["muriscvnn.inc_dir"]
            cmakeArgs.append("-DTFLM_OPTIMIZED_KERNEL=cmsis_nn")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={muriscvnnLib}")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={muriscvnnInc}")
        elif cmsisnn:
            flags__ = utils.makeFlags((True, arch), (dbg, "dbg"))
            cmsisnnLib = context.cache["cmsisnn.lib", flags__]
            cmsisDir = Path(context.cache["cmsisnn.dir"])
            cmsisIncs = r"\;".join(
                [
                    str(cmsisDir),
                    str(cmsisDir / "CMSIS" / "Core" / "Include"),
                    str(cmsisDir / "CMSIS" / "NN" / "Include"),
                    str(cmsisDir / "CMSIS" / "DSP" / "Include"),
                ]
            )
            cmakeArgs.append("-DTFLM_OPTIMIZED_KERNEL=cmsis_nn")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_LIB={cmsisnnLib}")
            cmakeArgs.append(f"-DTFLM_OPTIMIZED_KERNEL_INCLUDE_DIR={cmsisIncs}")
        utils.mkdirs(tflmcBuildDir)
        # utils.cmake("-DTF_SRC=" + str(tfSrcDir), str(tflmcSrcDir), debug=params["dbg"], cwd=tflmcBuildDir)
        utils.cmake(
            *cmakeArgs,
            str(tflmcSrcDir),
            debug=dbg,
            cwd=tflmcBuildDir,
            live=verbose,
        )
        utils.make(cwd=tflmcBuildDir, threads=threads, live=verbose)
        utils.mkdirs(tflmcInstallDir)
        utils.move(tflmcBuildDir / "compiler", tflmcExe)
    context.cache["tflmc.build_dir", flags] = tflmcBuildDir
    context.cache["tflmc.exe", flags] = tflmcExe
    context.export_paths.add(tflmcExe.parent)
