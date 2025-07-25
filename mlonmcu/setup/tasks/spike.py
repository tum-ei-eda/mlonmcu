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
import shutil
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger
from mlonmcu.config import str2bool

from .common import get_task_factory
from .ara import _validate_ara_rtl

logger = get_logger()

Tasks = get_task_factory()


def _validate_spike(context: MlonMcuContext, params=None):
    if not context.environment.has_target("spike") and not _validate_ara_rtl(context, params=params):
        return False
    user_vars = context.environment.vars
    if "spike.exe" not in user_vars:  # TODO: also check command line flags?
        assert "spike" in context.environment.repos, "Undefined repository: 'spike'"
    return True


def _validate_spikepk(context: MlonMcuContext, params=None):
    if not _validate_spike(context, params=params):
        return False
    user_vars = context.environment.vars
    enable_rv32 = str2bool(user_vars.get("spikepk.enable_rv32", True))
    enable_rv64 = str2bool(user_vars.get("spikepk.enable_rv64", False))
    if (enable_rv32 and "spike.pk_rv32" not in user_vars) or (
        enable_rv64 and "spike.pk_rv64" not in user_vars
    ):  # TODO: also check command line flags?
        assert "spikepk" in context.environment.repos, "Undefined repository: 'spikepk'"
    return True


def _validate_spikepk_build(context: MlonMcuContext, params=None):
    if not _validate_spike(context, params=params):
        return False
    if not _validate_spikepk(context, params=params):
        return False
    user_vars = context.environment.vars
    enable_rv32 = str2bool(user_vars.get("spikepk.enable_rv32", True))
    enable_rv64 = str2bool(user_vars.get("spikepk.enable_rv64", False))
    assert params is not None
    xlen = params.get("xlen")
    if xlen == 32 and not enable_rv32:
        return False
    if xlen == 64 and not enable_rv64:
        return False
    return True


def _validate_spike_clean(context: MlonMcuContext, params={}):
    if not _validate_spike(context, params=params):
        return False
    user_vars = context.environment.vars
    keep_build_dir = user_vars.get("spike.keep_build_dir", True)
    return not keep_build_dir


@Tasks.provides(["spikepk.src_dir"])
@Tasks.validate(_validate_spikepk)
@Tasks.register(category=TaskType.TARGET)
def clone_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the spike proxy kernel."""
    spikepkName = utils.makeDirName("spikepk")
    spikepkSrcDir = context.environment.paths["deps"].path / "src" / spikepkName
    user_vars = context.environment.vars
    if "spike.pk" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(spikepkSrcDir):
        spikepkRepo = context.environment.repos["spikepk"]
        utils.clone_wrapper(spikepkRepo, spikepkSrcDir, refresh=rebuild)
    context.cache["spikepk.src_dir"] = spikepkSrcDir


@Tasks.needs(["spikepk.src_dir"])
@Tasks.optional(["riscv_gcc_rv{xlen}.install_dir", "riscv_gcc_rv{xlen}.name"])
@Tasks.provides(["spikepk_rv{xlen}.build_dir", "spikepk_rv{xlen}.install_dir", "spike.pk_rv{xlen}"])
# TODO: allow arch,abi
# @Tasks.param("arch", ["default"])  # ["rv32gc", "rv64gc", "rv32im", "rv64im"]
@Tasks.param("xlen", [32, 64])
@Tasks.validate(_validate_spikepk_build)
@Tasks.register(category=TaskType.TARGET)
def build_spike_pk(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build Spike proxy kernel."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    # default_arch = "rv32gc"
    # arch = params.get("arch", "rv32gc")
    xlen = params["xlen"]
    assert xlen in [32, 64]
    spikepkName = utils.makeDirName(f"spikepk_rv{xlen}")
    spikepkSrcDir = context.cache["spikepk.src_dir"]
    spikepkBuildDir = context.environment.paths["deps"].path / "build" / spikepkName
    spikepkInstallDir = context.environment.paths["deps"].path / "install" / spikepkName
    if "spike.pk_rv{xlen}" in user_vars:  # TODO: also check command line flags?
        return False
    arch = user_vars.get(
        f"spikepk_rv{xlen}.default_arch", "rv32imafdc_zifencei_zicsr" if xlen == 32 else "rv64imafdc_zifencei_zicsr"
    )
    abi = user_vars.get(f"spikepk_rv{xlen}.default_abi", "ilp32d" if xlen == 32 else "lp64d")
    # spikepkBin = spikepkInstallDir / f"pk_{arch}_{abi}"
    spikepkDefaultBin = spikepkInstallDir / "pk"
    # if rebuild or not (utils.is_populated(spikepkBuildDir) and spikepkBin.is_file()):
    if rebuild or not (utils.is_populated(spikepkBuildDir) and spikepkDefaultBin.is_file()):
        # No need to build a vext and non-vext variant?
        utils.mkdirs(spikepkBuildDir)
        gccName = context.cache[f"riscv_gcc_rv{xlen}.name"]
        # assert gccName == "riscv32-unknown-elf", "Spike PK requires a non-multilib toolchain!"
        if f"riscv_gcc_rv{xlen}.install_dir" in user_vars:
            riscv_gcc = user_vars[f"riscv_gcc_rv{xlen}.install_dir"]
        else:
            riscv_gcc = context.cache[f"riscv_gcc_rv{xlen}.install_dir"]
        spikepkArgs = []
        spikepkArgs.append(f"--with-arch={arch}")
        spikepkArgs.append(f"--with-abi={abi}")
        spikepkArgs.append("--prefix=" + str(riscv_gcc))
        spikepkArgs.append("--host=" + gccName)
        env = os.environ.copy()
        env["PATH"] = str(Path(riscv_gcc) / "bin") + ":" + env["PATH"]
        utils.execute(
            str(spikepkSrcDir / "configure"),
            *spikepkArgs,
            cwd=spikepkBuildDir,
            env=env,
            live=False,
        )
        utils.make(cwd=spikepkBuildDir, threads=threads, live=verbose, env=env)
        # utils.make(target="install", cwd=spikepkBuildDir, live=verbose, env=env)
        utils.mkdirs(spikepkInstallDir)
        # utils.move(spikepkBuildDir / "pk", spikepkBin)
        # if arch == default_arch:
        utils.copy(spikepkBuildDir / "pk", spikepkDefaultBin)
    context.cache[f"spikepk_rv{xlen}.build_dir"] = spikepkBuildDir
    context.cache[f"spikepk_rv{xlen}.install_dir"] = spikepkInstallDir
    # if arch == default_arch:
    context.cache[f"spike.pk_rv{xlen}"] = spikepkDefaultBin
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
        utils.clone_wrapper(spikeRepo, spikeSrcDir, refresh=rebuild)
    context.cache["spike.src_dir"] = spikeSrcDir


@Tasks.needs(["spike.src_dir"])
@Tasks.optional(["boost.install_dir"])
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
    boostDir = context.cache.get("boost.install_dir")
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
        # spikeArgs.append("--prefix=" + str(context.cache["riscv_gcc.install_dir"]))
        spikeArgs.append("--prefix=" + str(spikeInstallDir))
        spikeArgs.append("--enable-misaligned")
        if boostDir is not None:
            spikeArgs.append(f"--with-boost={boostDir}")
        utils.execute(
            str(Path(spikeSrcDir) / "configure"),
            *spikeArgs,
            cwd=spikeBuildDir,
            live=False,
        )
        utils.make(cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.make("install", cwd=spikeBuildDir, threads=threads, live=verbose)
        utils.mkdirs(spikeInstallDir)
        utils.move(spikeBuildDir / "spike", spikeExe)
    context.cache["spike.build_dir"] = spikeBuildDir
    context.cache["spike.install_dir"] = spikeInstallDir
    context.cache["spike.exe"] = spikeExe
    context.export_paths.add(spikeInstallDir)


@Tasks.needs(["spike.exe", "spike.build_dir"])  # TODO: make sure spike.exe has beeen copies before
@Tasks.removes(["spike.build_dir"])  # TODO: implement
@Tasks.validate(_validate_spike_clean)
@Tasks.register(category=TaskType.TARGET)
def clean_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Cleanup Spike build dir."""
    spikeBuildDir = context.cache["spike.build_dir"]
    shutil.rmtree(spikeBuildDir)
    del context.cache["spike.build_dir"]


def _validate_microtvm_spike(context: MlonMcuContext, params=None):
    return context.environment.has_target("microtvm_spike")


@Tasks.provides(["microtvm_spike.src_dir", "microtvm_spike.template"])
@Tasks.validate(_validate_microtvm_spike)
@Tasks.register(category=TaskType.TARGET)
def clone_microtvm_spike(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the microtvm-spike-template repository."""
    name = utils.makeDirName("microtvm_spike")
    srcDir = context.environment.paths["deps"].path / "src" / name
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["microtvm_spike"]
        utils.clone_wrapper(repo, srcDir, refresh=rebuild)
    context.cache["microtvm_spike.src_dir"] = srcDir
    context.cache["microtvm_spike.template"] = srcDir / "template_project"
