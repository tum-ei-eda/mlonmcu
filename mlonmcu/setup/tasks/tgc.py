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


def _validate_tgc(context: MlonMcuContext, params=None):
    if not context.environment.has_target("tgc"):
        return False
    user_vars = context.environment.vars
    if "tgc.exe" not in user_vars:  # TODO: also check command line flags?
        assert "spike" in context.environment.repos, "Undefined repository: 'spike'"
    return True

@Tasks.provides(["tgc.src_dir"])
@Tasks.validate(_validate_tgc)
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
        logger.debug(context.environment.repos)
        tgcRepo = context.environment.repos["tgc"]
        utils.clone(tgcRepo.url, tgcSrcDir, branch="develop", recursive=True)
        #utils.move("/home/gabriel/.config/mlonmcu/environments/iss_gen/deps/src/tgc/dbt-rise-tgc/src-gen","/home/gabriel/.config/mlonmcu/environments/iss_gen/deps/src/tgc/dbt-rise-tgc/src-not-gen")
        #shutil.copytree("/scratch/gabriel/TGC-ISS/dbt-rise-tgc/src-gen/","/home/gabriel/.config/mlonmcu/environments/iss_gen/deps/src/tgc/dbt-rise-tgc/src-gen")
        if "tgc_gen" in context.environment.repos:
            tgc_gen_src_dir = context.environment.paths["deps"].path / "src" / "tgc_gen"
            tgc_repo_extra = context.environment.repos["tgc_gen"]
            utils.clone(tgc_repo_extra.url, tgc_gen_src_dir, branch="develop", recursive=True )
            context.cache["tgc.gen_src_dir"] = tgc_gen_src_dir
    
    context.cache["tgc.src_dir"] = tgcSrcDir

    
"""
@Tasks.needs(["tgc.gen_src_dir", "tgc.src_dir"])
@Tasks.register(category=TaskType.TARGET)
def generate_tgc_cores(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count() 
):
    tgc_gen_dir = context.cache["tgc.gen_src_dir"]
    tgc_src_dir = context.cache["tgc.src_dir"]
    backends = ["interp", "asmjit"]
    versions = ["TGC5A","TGC5B","TGC5C"]

    utils.exec_getout(
            "direnv allow",
            cwd=tgc_src_dir,
            shell=True,
            executable="/bin/bash",
        )

    for backend in backends:
            for version in versions:
                utils.exec_getout(
                    f"direnv exec {tgc_src_dir} {tgc_src_dir}/../tgc_gen/scripts/generate_iss.sh -c {version} -o {tgc_src_dir}/dbt-rise-tgc/ -b {backend} {tgc_src_dir}/../tgc_gen/CoreDSL/{version}.core_desc",
                    cwd=tgc_src_dir,
                    live=False,
                    print_output=True,
                )
"""
@Tasks.needs(["tgc.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.validate(_validate_tgc)
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
    backends = ["interp", "asmjit"]
    versions = ["TGC5A","TGC5B"]
    if "tgc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(tgcBuildDir) and tgcExe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(tgcBuildDir)
        if "tgc_gen" in context.environment.repos:
            utils.exec_getout(
                "direnv allow",
                cwd=tgcSrcDir,
                shell=True,
                executable="/bin/bash",
            )
            for backend in backends:
                for version in versions:
                    cmd = f'/bin/bash -c "source ~/.sdkman/bin/sdkman-init.sh && sdk use java 11.0.21-tem && java --version && {tgcSrcDir}/../tgc_gen/scripts/generate_iss.sh -c {version} -o dbt-rise-tgc/ -b {backend} {tgcSrcDir}/../tgc_gen/CoreDSL/{version}.core_desc"'
                    utils.exec_getout(
                        cmd,
                        cwd=tgcSrcDir,
                        print_output=True,
                        shell=True,
                    )
        """
        utils.exec_getout(
            "direnv allow",
            cwd=tgcSrcDir,
            shell=True,
            executable="/bin/bash",
        )
        for backend in backends:
            for version in versions:
                utils.exec_getout(
                    f"{tgcSrcDir}/../tgc_gen/scripts/generate_iss.sh -c {version} -o {tgcSrcDir}/dbt-rise-tgc/ -b {backend} {tgcSrcDir}/../tgc_gen/CoreDSL/{version}.core_desc",
                    cwd=tgcSrcDir,
                    print_output=True,
                )
                
                command = f"direnv exec {tgcSrcDir} sdk use java 11.0.21-tem && {tgcSrcDir}/../TGC-GEN/scripts/generate_iss.sh -c {version} -o {tgcSrcDir}/dbt-rise-tgc/ -b {backend} {tgcSrcDir}/../TGC-GEN/CoreDSL/{version}.core_desc"
                output = subprocess.run(
                    ["bash", "-l", "-c", command],
                    cwd=tgcSrcDir,
                    capture_output=True,                    
                )
                logger.debug(output)"""
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
