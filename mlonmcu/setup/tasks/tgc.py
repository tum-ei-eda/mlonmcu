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


@Tasks.provides(["tgc.src_dir"])
@Tasks.register(category=TaskType.TARGET)
def clone_tgc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the tgc simulator."""
    tgcName = utils.makeDirName("tgc")
    tgcSrcDir = context.environment.paths["deps"].path / "src" / tgcName
    user_vars = context.environment.vars
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(tgcSrcDir):

        tgcRepo = context.environment.repos["tgc"]
        utils.clone(tgcRepo.url, tgcSrcDir, branch=tgcRepo.ref, recursive=True)
    context.cache["tgc.src_dir"] = tgcSrcDir


@Tasks.needs(["tgc.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.provides(["tgc.build_dir", "tgc.exe"])
@Tasks.register(category=TaskType.TARGET)
def build_tgc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    #Build tgc simulator.
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    tgcName = utils.makeDirName("tgc")
    tgcSrcDir = context.cache["tgc.src_dir"]
    tgcBuildDir = context.environment.paths["deps"].path / "build" / tgcName
    tgcInstallDir = context.environment.paths["deps"].path / "install" / tgcName
    tgcExe = tgcInstallDir / "tgc-sim"
    user_vars = context.environment.vars
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(tgcBuildDir) and tgcExe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(tgcBuildDir)
        """
        profile = utils.exec_getout("conan", "profile", "list",)
        if profile != "default":
            utils.exec_getout(
                "conan", "profile", "new", "default", "--detect",
                cwd=tgcBuildDir,
                live=False,
                print_output=True,
            )
            utils.exec_getout(
                "conan", "profile", "update", "settings.compiler.libcxx=libstdc++11", "default",
                cwd=tgcBuildDir,
                live=False,
                print_output=True,
            )
            
        utils.exec_getout(
            "module", "load", "./Modulefile",
            cwd=tgcSrcDir,
            live=False,
            print_output=True,
        )  """      
        utils.exec_getout(
            "cmake", "-S", tgcSrcDir, "-B", ".",
            cwd=tgcBuildDir,
            live=False,
            print_output=True,
        )        
        utils.make(cwd=tgcBuildDir, threads=threads, live=verbose)
        # utils.make(target="install", cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.mkdirs(tgcInstallDir)
        utils.move(tgcBuildDir / "dbt-rise-tgc" / "tgc-sim", tgcExe)
    context.cache["tgc.build_dir"] = tgcBuildDir
    context.cache["tgc.exe"] = tgcExe
