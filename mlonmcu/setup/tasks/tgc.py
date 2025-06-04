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

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_tgc(context: MlonMcuContext, params=None):
    del params
    if not context.environment.has_target("tgc"):
        return False
    user_vars = context.environment.vars
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    # assert "tgc" in context.environment.repos, "Undefined repository: 'tgc'"
    return True


def _validate_tgc_gen(context: MlonMcuContext, params=None):
    if not _validate_tgc(context, params=params):
        return False
    user_vars = context.environment.vars
    enable = user_vars.get("tgc.gen_enable", False)
    return enable


@Tasks.provides(["tgc.src_dir"])
@Tasks.validate(_validate_tgc)
@Tasks.register(category=TaskType.TARGET)
def clone_tgc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the tgc simulator."""
    del params, verbose, threads
    tgcName = utils.makeDirName("tgc")
    tgcSrcDir = context.environment.paths["deps"].path / "src" / tgcName
    user_vars = context.environment.vars
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(tgcSrcDir):
        tgcRepo = context.environment.repos["tgc"]
        utils.clone_wrapper(tgcRepo, tgcSrcDir)

    context.cache["tgc.src_dir"] = tgcSrcDir


@Tasks.provides(["tgc_bsp.src_dir"])
@Tasks.validate(_validate_tgc)
@Tasks.register(category=TaskType.TARGET)
def clone_tgc_bsp(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the tgc BSP."""
    del params, verbose, threads
    bspName = utils.makeDirName("tgc_bsp")
    bspSrcDir = context.environment.paths["deps"].path / "src" / bspName
    user_vars = context.environment.vars
    if "tgc_bsp.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(bspSrcDir):
        bspRepo = context.environment.repos["tgc_bsp"]
        utils.clone_wrapper(bspRepo, bspSrcDir)

    context.cache["tgc_bsp.src_dir"] = bspSrcDir


@Tasks.provides(["tgc.gen_src_dir"])
@Tasks.validate(_validate_tgc_gen)
@Tasks.register(category=TaskType.TARGET)
def clone_tgc_gen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the tgc generator."""
    del params, verbose, threads
    tgcGenName = utils.makeDirName("tgc_gen")
    tgcGenSrcDir = context.environment.paths["deps"].path / "src" / tgcGenName
    # user_vars = context.environment.vars
    if rebuild or not utils.is_populated(tgcGenSrcDir):
        tgcGenRepo = context.environment.repos["tgc_gen"]
        utils.clone_wrapper(tgcGenRepo, tgcGenSrcDir)
    context.cache["tgc.gen_src_dir"] = tgcGenSrcDir


@Tasks.needs(["tgc.src_dir"])
@Tasks.optional(["tgc.gen_src_dir"])
@Tasks.validate(_validate_tgc)
@Tasks.provides(["tgc.build_dir", "tgc.exe"])
@Tasks.register(category=TaskType.TARGET)
def build_tgc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    # Build tgc simulator.
    if not params:
        params = {}
    user_vars = context.environment.vars
    gen_enable = user_vars.get("tgc.gen_enable", False)
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    tgcName = utils.makeDirName("tgc")
    tgcSrcDir = context.cache["tgc.src_dir"]
    tgcBuildDir = context.environment.paths["deps"].path / "build" / tgcName
    tgcInstallDir = context.environment.paths["deps"].path / "install" / tgcName
    tgcExe = tgcInstallDir / "tgc-sim"
    user_vars = context.environment.vars
    # backends = ["interp", "asmjit"]
    # versions = ["TGC5A", "TGC5B"]
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(tgcBuildDir) and tgcExe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(tgcBuildDir)
        if "tgc_gen" in context.environment.repos:
            utils.execute(
                "direnv allow",
                cwd=tgcSrcDir,
                shell=True,
                executable="/bin/bash",
            )
            if gen_enable:
                raise NotImplementedError
                # tgcGenSrcDir = context.cache["tgc.gen_src_dir"]
                # for backend in backends:
                #     for version in versions:
                #         cmd = f'/bin/bash -c "source ~/.sdkman/bin/sdkman-init.sh && sdk use java 11.0.21-tem && java
                # --version && {tgcSrcDir}/../tgc_gen/scripts/generate_iss.sh -c {version} -o dbt-rise-tgc/ -b {backend}
                # {tgcSrcDir}/../tgc_gen/CoreDSL/{version}.core_desc"'
                #         utils.execute(
                #             cmd,
                #             cwd=tgcSrcDir,
                #             shell=True,
                #         )
        utils.execute(
            "cmake",
            "-S",
            tgcSrcDir,
            "-B",
            ".",
            cwd=tgcBuildDir,
            live=verbose,
        )
        utils.make(cwd=tgcBuildDir, threads=threads, live=verbose)
        # utils.make(target="install", cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.mkdirs(tgcInstallDir)
        utils.move(tgcBuildDir / "dbt-rise-tgc" / "tgc-sim", tgcExe)
    context.cache["tgc.build_dir"] = tgcBuildDir
    context.cache["tgc.exe"] = tgcExe
