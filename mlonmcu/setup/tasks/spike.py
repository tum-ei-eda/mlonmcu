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


def _validate_spike(context: MlonMcuContext, params=None):
    if not context.environment.has_target("spike"):
        return False
    if params.get("vext", False):
        if params.get("pext", False):
            return False  # Can not use booth at a time
        if not context.environment.supports_feature("vext"):
            return False
    if params.get("pext", False):
        if params.get("vext", False):
            return False  # Can not use booth at a time
        if not context.environment.supports_feature("pext"):
            return False
    user_vars = context.environment.vars
    if "spike.pk" not in user_vars:  # TODO: also check command line flags?
        assert "spikepk" in context.environment.repos, "Undefined repository: 'spikepk'"
    if "spike.exe" not in user_vars:  # TODO: also check command line flags?
        assert "spike" in context.environment.repos, "Undefined repository: 'spike'"
    return True


@Tasks.provides(["spikepk.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the spike proxt kernel."""
    spikepkName = utils.makeDirName("spikepk")
    spikepkSrcDir = context.environment.paths["deps"].path / "src" / spikepkName
    user_vars = context.environment.vars
    if "spike.pk" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(spikepkSrcDir):
        spikepkRepo = context.environment.repos["spikepk"]
        utils.clone(spikepkRepo.url, spikepkSrcDir, branch=spikepkRepo.ref)
    context.cache["spikepk.src_dir"] = spikepkSrcDir


@Tasks.needs(["spikepk.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.provides(["spikepk.build_dir", "spike.pk"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def build_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build Spike proxy kernel."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "spike.pk" in user_vars:  # TODO: also check command line flags?
        return False
    spikepkName = utils.makeDirName("spikepk")
    spikepkSrcDir = context.cache["spikepk.src_dir"]
    spikepkBuildDir = context.environment.paths["deps"].path / "build" / spikepkName
    spikepkInstallDir = context.environment.paths["deps"].path / "install" / spikepkName
    spikepkBin = spikepkInstallDir / "pk"
    if rebuild or not (utils.is_populated(spikepkBuildDir) and spikepkBin.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(spikepkBuildDir)
        gccName = context.cache["riscv_gcc.name"]
        # assert gccName == "riscv32-unknown-elf", "Spike PK requires a non-multilib toolchain!"
        vext = params.get("vext", False)
        pext = params.get("pext", False)
        assert not (pext and vext), "Currently only p or vector extension can be enabled at a time."
        if vext and "riscv_gcc.install_dir_vext" in user_vars:
            riscv_gcc = user_vars["riscv_gcc.install_dir_vext"]
        elif pext and "riscv_gcc.install_dir_pext" in user_vars:
            riscv_gcc = user_vars["riscv_gcc.install_dir_pext"]
        elif "riscv_gcc.install_dir" in user_vars:
            riscv_gcc = user_vars["riscv_gcc.install_dir"]
        else:
            riscv_gcc = context.cache["riscv_gcc.install_dir"]
        arch = "rv32imafdc"
        spikepkArgs = []
        spikepkArgs.append("--prefix=" + str(riscv_gcc))
        spikepkArgs.append("--host=" + gccName)
        spikepkArgs.append(f"--with-arch={arch}")
        spikepkArgs.append("--with-abi=ilp32d")
        env = os.environ.copy()
        env["PATH"] = str(Path(riscv_gcc) / "bin") + ":" + env["PATH"]
        utils.exec_getout(
            str(spikepkSrcDir / "configure"),
            *spikepkArgs,
            cwd=spikepkBuildDir,
            env=env,
            live=False,
            print_output=False,
        )
        utils.make(cwd=spikepkBuildDir, threads=threads, live=verbose, env=env)
        # utils.make(target="install", cwd=spikepkBuildDir, live=verbose, env=env)
        utils.mkdirs(spikepkInstallDir)
        utils.move(spikepkBuildDir / "pk", spikepkBin)
    context.cache["spikepk.build_dir"] = spikepkBuildDir
    context.cache["spike.pk"] = spikepkBin
    context.export_paths.add(spikepkInstallDir)


@Tasks.provides(["spike.src_dir"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the spike simulator."""
    spikeName = utils.makeDirName("spike")
    spikeSrcDir = context.environment.paths["deps"].path / "src" / spikeName
    user_vars = context.environment.vars
    if "spike.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(spikeSrcDir):
        spikeRepo = context.environment.repos["spike"]
        utils.clone(spikeRepo.url, spikeSrcDir, branch=spikeRepo.ref)
    context.cache["spike.src_dir"] = spikeSrcDir


@Tasks.needs(["spike.src_dir", "riscv_gcc.install_dir", "riscv_gcc.name"])
@Tasks.provides(["spike.build_dir", "spike.exe"])
@Tasks.validate(_validate_spike)
@Tasks.register(category=TaskType.TARGET)
def build_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build Spike simulator."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "spike.exe" in user_vars:  # TODO: also check command line flags?
        return False
    spikeName = utils.makeDirName("spike")
    spikeSrcDir = context.cache["spike.src_dir"]
    spikeBuildDir = context.environment.paths["deps"].path / "build" / spikeName
    spikeInstallDir = context.environment.paths["deps"].path / "install" / spikeName
    spikeExe = spikeInstallDir / "spike"
    user_vars = context.environment.vars
    if "spike.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not (utils.is_populated(spikeBuildDir) and spikeExe.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(spikeBuildDir)
        spikeArgs = []
        spikeArgs.append("--prefix=" + str(context.cache["riscv_gcc.install_dir"]))
        utils.exec_getout(
            str(spikeSrcDir / "configure"),
            *spikeArgs,
            cwd=spikeBuildDir,
            live=False,
            print_output=False,
        )
        utils.make(cwd=spikeBuildDir, threads=threads, live=verbose)
        # utils.make(target="install", cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.mkdirs(spikeInstallDir)
        utils.move(spikeBuildDir / "spike", spikeExe)
    context.cache["spike.build_dir"] = spikeBuildDir
    context.cache["spike.exe"] = spikeExe
    context.export_paths.add(spikeInstallDir)
