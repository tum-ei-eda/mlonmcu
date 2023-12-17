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

from .tvm import _validate_tvm
from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_utvmcg(context: MlonMcuContext, params=None):
    if not _validate_tvm(context, params=params):
        return False
    return context.environment.has_backend("tvmcg")


@Tasks.provides(["utvmcg.src_dir"])
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def clone_utvm_staticrt_codegen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the uTVM code generator."""
    utvmcgName = utils.makeDirName("utvmcg")
    utvmcgSrcDir = context.environment.paths["deps"].path / "src" / utvmcgName
    if rebuild or not utils.is_populated(utvmcgSrcDir):
        utvmcgRepo = context.environment.repos["utvm_staticrt_codegen"]
        utils.clone_wrapper(utvmcgRepo, utvmcgSrcDir, refresh=rebuild)
    context.cache["utvmcg.src_dir"] = utvmcgSrcDir


@Tasks.needs(["utvmcg.src_dir", "tvm.src_dir"])
@Tasks.provides(["utvmcg.build_dir", "utvmcg.exe"])
@Tasks.param("dbg", False)
@Tasks.validate(_validate_utvmcg)
@Tasks.register(category=TaskType.BACKEND)
def build_utvm_staticrt_codegen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the uTVM code generator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    utvmcgName = utils.makeDirName("utvmcg", flags=flags)
    utvmcgSrcDir = context.cache["utvmcg.src_dir"]
    utvmcgBuildDir = context.environment.paths["deps"].path / "build" / utvmcgName
    utvmcgInstallDir = context.environment.paths["deps"].path / "install" / utvmcgName
    utvmcgExe = utvmcgInstallDir / "compiler"
    if rebuild or not utils.is_populated(utvmcgSrcDir) or not utvmcgExe.is_file():
        utvmcgArgs = []
        utvmcgArgs.append("-DTVM_DIR=" + str(context.cache["tvm.src_dir"]))
        crtConfigPath = context.cache["tvm.src_dir"] / "apps" / "bundle_deploy" / "crt_config"
        if context:
            user_vars = context.environment.vars
            if "tvm.crt_config_dir" in user_vars:
                crtConfigPath = Path(user_vars["tvm.crt_config_dir"])
        utvmcgArgs.append("-DTVM_CRT_CONFIG_DIR=" + str(crtConfigPath))
        utils.mkdirs(utvmcgBuildDir)
        utils.cmake(
            utvmcgSrcDir,
            *utvmcgArgs,
            cwd=utvmcgBuildDir,
            debug=params["dbg"],
            live=verbose,
        )
        utils.make(cwd=utvmcgBuildDir, threads=threads, live=verbose)
        utils.mkdirs(utvmcgInstallDir)
        utils.move(utvmcgBuildDir / "utvm_staticrt_codegen", utvmcgExe)
    context.cache["utvmcg.build_dir", flags] = utvmcgBuildDir
    context.cache["utvmcg.exe", flags] = utvmcgExe
    context.export_paths.add(utvmcgExe.parent)
