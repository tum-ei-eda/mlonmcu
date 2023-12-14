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


def _validate_etiss(context: MlonMcuContext, params={}):
    if "dbg" in params:
        dbg = params["dbg"]
        if dbg:
            if not context.environment.has_feature("etissdbg"):
                return False
    return context.environment.has_target("etiss_pulpino")


@Tasks.provides(["etiss.src_dir"])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def clone_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the ETISS repository."""
    etissName = utils.makeDirName("etiss")
    user_vars = context.environment.vars
    if "etiss.src_dir" in user_vars:
        etissSrcDir = Path(user_vars["etiss.src_dir"])
        rebuild = False
    else:
        etissSrcDir = context.environment.paths["deps"].path / "src" / etissName
    if rebuild or not utils.is_populated(etissSrcDir):
        etissRepo = context.environment.repos["etiss"]
        utils.clone(etissRepo.url, etissSrcDir, branch=etissRepo.ref)
    context.cache["etiss.src_dir"] = etissSrcDir


@Tasks.needs(["etiss.src_dir", "llvm.install_dir"])
@Tasks.provides(["etiss.build_dir", "etiss.install_dir"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def build_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the ETISS simulator."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.environment.paths["deps"].path / "build" / etissName
    etissInstallDir = context.environment.paths["deps"].path / "install" / etissName
    llvmInstallDir = context.cache["llvm.install_dir"]
    user_vars = context.environment.vars
    if "etiss.build_dir" in user_vars or "etiss.install_dir" in user_vars:
        return False
    if rebuild or not utils.is_populated(etissBuildDir):
        utils.mkdirs(etissBuildDir)
        env = os.environ.copy()
        env["LLVM_DIR"] = str(llvmInstallDir)
        utils.cmake(
            context.cache["etiss.src_dir"],
            "-DCMAKE_INSTALL_PREFIX=" + str(etissInstallDir),
            cwd=etissBuildDir,
            debug=params["dbg"],
            env=env,
            live=verbose,
        )
        utils.make(cwd=etissBuildDir, threads=threads, live=verbose)
    context.cache["etiss.build_dir", flags] = etissBuildDir
    context.cache["etiss.install_dir", flags] = etissInstallDir


@Tasks.needs(["etiss.build_dir"])
@Tasks.provides(["etiss.install_dir", "etissvp.exe", "etissvp.script"])
@Tasks.param("dbg", [False, True])
@Tasks.validate(_validate_etiss)
@Tasks.register(category=TaskType.TARGET)
def install_etiss(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Install ETISS."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "etiss.install_dir" in user_vars and "etissvp.exe" in user_vars and "etissvp.script" in user_vars:
        return False
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # etissName = utils.makeDirName("etiss", flags=flags)
    etissBuildDir = context.cache["etiss.build_dir", flags]
    etissInstallDir = context.cache["etiss.install_dir", flags]
    etissvpExe = etissInstallDir / "bin" / "bare_etiss_processor"
    etissvpScript = etissInstallDir / "bin" / "run_helper.sh"
    if rebuild or not utils.is_populated(etissInstallDir) or not etissvpExe.is_file():
        utils.make("install", cwd=etissBuildDir, threads=threads, live=verbose)
    context.cache["etiss.install_dir", flags] = etissInstallDir
    context.cache["etissvp.exe", flags] = etissvpExe
    context.cache["etissvp.script", flags] = etissvpScript


def _validate_microtvm_etissvp(context: MlonMcuContext, params=None):
    return context.environment.has_target("microtvm_etissvp")


@Tasks.provides(["microtvm_etissvp.src_dir", "microtvm_etissvp.template"])
@Tasks.validate(_validate_microtvm_etissvp)
@Tasks.register(category=TaskType.FEATURE)
def clone_microtvm_etissvp(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the microtvm-etissvp-template repository."""
    name = utils.makeDirName("microtvm_etissvp")
    srcDir = context.environment.paths["deps"].path / "src" / name
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["microtvm_etissvp"]
        utils.clone(repo.url, srcDir, branch=repo.ref, refresh=rebuild)
    context.cache["microtvm_etissvp.src_dir"] = srcDir
    context.cache["microtvm_etissvp.template"] = srcDir / "template_project"
