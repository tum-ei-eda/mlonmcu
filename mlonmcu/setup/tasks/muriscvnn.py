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


def _validate_muriscvnn(context: MlonMcuContext, params=None):
    if not context.environment.supports_feature("muriscvnn"):
        return False
    user_vars = context.environment.vars
    if "muriscvnn.src_dir" not in user_vars:
        assert "muriscvnn" in context.environment.repos, "Undefined repository: 'muriscvnn'"
    if params:
        toolchain = params.get("toolchain", "gcc")
        if not context.environment.has_toolchain(toolchain):
            return False
        target_arch = params.get("target_arch", "riscv")
        if target_arch == "riscv":
            if params.get("vext", False):
                if not context.environment.supports_feature("vext"):
                    return False
            if params.get("pext", False):
                if toolchain == "llvm":
                    # Unsupported
                    return False
                if not context.environment.supports_feature("pext"):
                    return False
            if params.get("vext", False) and params.get("pext", False):
                # Either pext or vext!
                return False
        elif target_arch == "x86":
            if toolchain != "gcc":
                return False
            if params.get("vext", False) or params.get("pext", False):
                return False
            # TODO: validate chosen toolchain?
    return True


def _validate_muriscvnn_build(context: MlonMcuContext, params=None):
    if not _validate_muriscvnn(context, params=params):
        return False
    user_vars = context.environment.vars
    skip_build = user_vars.get("muriscvnn.skip_build", True)
    # TODO: str2bool
    if skip_build:
        return False
    return True


@Tasks.provides(["muriscvnn.src_dir", "muriscvnn.inc_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the muRISCV-NN project."""
    muriscvnnName = utils.makeDirName("muriscvnn")
    user_vars = context.environment.vars
    # if "muriscvnn.lib" in user_vars:  # TODO: also check command line flags?
    #     return False
    if "muriscvnn.src_dir" in user_vars:
        muriscvnnSrcDir = user_vars["muriscvnn.src_dir"]
        muriscvnnIncludeDir = Path(muriscvnnSrcDir) / "Include"
        rebuild = False
    else:
        muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
        muriscvnnIncludeDir = muriscvnnSrcDir / "Include"
    if rebuild or not utils.is_populated(muriscvnnSrcDir):
        muriscvnnRepo = context.environment.repos["muriscvnn"]
        utils.clone_wrapper(muriscvnnRepo, muriscvnnSrcDir, refresh=rebuild)
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir
    context.cache["muriscvnn.inc_dir"] = muriscvnnIncludeDir


@Tasks.needs(["muriscvnn.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
# @Tasks.optional(["riscv_gcc.install_dir", "riscv_gcc.name", "arm_gcc.install_dir"])
@Tasks.provides(["muriscvnn.build_dir", "muriscvnn.lib"])
# @Tasks.param("dbg", [False, True])
@Tasks.param("dbg", [False])  # disable due to bug with vext gcc
@Tasks.param("vext", [False])
@Tasks.param("pext", [False])
@Tasks.param("toolchain", ["gcc"])
@Tasks.param("target_arch", ["x86", "riscv"])
@Tasks.validate(_validate_muriscvnn_build)
@Tasks.register(category=TaskType.OPT)
def build_muriscvnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build muRISCV-NN."""
    if not params:
        params = {}
    flags = utils.makeFlags(
        (params["dbg"], "dbg"),
        (params["vext"], "vext"),
        (params["pext"], "pext"),
        (True, params["toolchain"]),
        (True, params["target_arch"]),
    )
    flags_ = utils.makeFlags((params["vext"], "vext"), (params["pext"], "pext"))
    muriscvnnName = utils.makeDirName("muriscvnn", flags=flags)
    muriscvnnSrcDir = context.cache["muriscvnn.src_dir"]
    muriscvnnBuildDir = context.environment.paths["deps"].path / "build" / muriscvnnName
    muriscvnnInstallDir = context.environment.paths["deps"].path / "install" / muriscvnnName
    muriscvnnLib = muriscvnnInstallDir / "libmuriscvnn.a"
    user_vars = context.environment.vars
    if "muriscvnn.lib" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(muriscvnnBuildDir) and muriscvnnLib.is_file()):
        utils.mkdirs(muriscvnnBuildDir)
        muriscvnnArgs = []
        target_arch = params.get("target_arch", "riscv")
        if target_arch == "riscv":
            gccName = context.cache["riscv_gcc.name", flags_]
            toolchain = params.get("toolchain", "gcc")
            assert (
                gccName == "riscv32-unknown-elf" or toolchain != "llvm"
            ), "muRISCV-NN requires a non-multilib toolchain!"
            if "riscv_gcc.install_dir" in user_vars:
                riscv_gcc = user_vars["riscv_gcc.install_dir"]
            else:
                riscv_gcc = context.cache["riscv_gcc.install_dir", flags_]
            muriscvnnArgs.append("-DRISCV_GCC_PREFIX=" + str(riscv_gcc))
            muriscvnnArgs.append("-DTOOLCHAIN=" + params["toolchain"].upper())
            vext = params.get("vext", False)
            pext = params.get("pext", False)
            muriscvnnArgs.append("-DUSE_VEXT=" + ("ON" if vext else "OFF"))
            muriscvnnArgs.append("-DUSE_PEXT=" + ("ON" if pext else "OFF"))
            muriscvnnArgs.append(f"-DRISCV_GCC_BASENAME={gccName}")
            arch = "rv32imafdc"
            if vext:
                arch += "v"
            if pext:
                arch += "p"
            muriscvnnArgs.append(f"-DRISCV_ARCH={arch}")
        elif target_arch == "x86":
            toolchain = params.get("toolchain", "gcc")
            muriscvnnArgs.append("-DTOOLCHAIN=x86")
            muriscvnnArgs.append("-DUSE_VEXT=OFF")
            muriscvnnArgs.append("-DUSE_PEXT=OFF")
        else:
            raise RuntimeError(f"Unsupported target_arch for muriscvnn: {target_arch}")
        utils.cmake(
            muriscvnnSrcDir,
            *muriscvnnArgs,
            cwd=muriscvnnBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=muriscvnnBuildDir, threads=threads, live=verbose)
        utils.mkdirs(muriscvnnInstallDir)
        utils.move(muriscvnnBuildDir / "Source" / "libmuriscvnn.a", muriscvnnLib)
    context.cache["muriscvnn.build_dir", flags] = muriscvnnBuildDir
    context.cache["muriscvnn.lib", flags] = muriscvnnLib
