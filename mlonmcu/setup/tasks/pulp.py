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
import pkg_resources
from pathlib import Path
import multiprocessing

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()
Tasks = get_task_factory()

#############
# pulp-gcc  #
#############


def _validate_pulp(context: MlonMcuContext, params=None):
    return context.environment.has_target("gvsoc_pulp") or context.environment.has_target("gvsoc_pulpissimo")


def _validate_pulp_gcc(context: MlonMcuContext, params=None):
    return context.environment.has_toolchain("gcc") and _validate_pulp(context, params)


@Tasks.provides(["pulp_gcc.install_dir", "pulp_gcc.name"])
@Tasks.validate(_validate_pulp_gcc)
@Tasks.register(category=TaskType.TOOLCHAIN)
def install_pulp_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install the PULP (RISCV) GCC toolchain."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    from_source = user_vars.get("pulp_gcc.from_source", False)
    flags = utils.makeFlags()
    pulpGccName = utils.makeDirName("pulp_gcc", flags=flags)
    pulpGccInstallDir = context.environment.paths["deps"].path / "install" / pulpGccName
    if "pulp_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        pulpGccInstallDir = user_vars["pulp_gcc.install_dir"]
    else:
        if rebuild or not utils.is_populated(pulpGccInstallDir):
            if from_source:
                pulpGccSrcDir = context.environment.paths["deps"].path / "src" / pulpGccName
                pulpGccBuildDir = context.environment.paths["deps"].path / "build" / pulpGccName
                utils.mkdirs(pulpGccBuildDir)
                if rebuild or not utils.is_populated(pulpGccSrcDir):
                    pulpGccRepo = context.environment.repos["pulp_gcc"]
                    utils.clone_wrapper(pulpGccRepo, pulpGccSrcDir, refresh=rebuild, recursive=True)
                env = os.environ.copy()
                env["PATH"] = str(pulpGccInstallDir) + ":" + env["PATH"]
                if rebuild or not utils.is_populated(pulpGccBuildDir):
                    pulpGccArgs = []
                    pulpGccArgs.append("--prefix=" + str(pulpGccInstallDir))
                    pulpGccArgs.append("--with-arch=rv32imc")
                    pulpGccArgs.append("--with-cmodel=medlow")
                    pulpGccArgs.append("--enable-multilib")
                    utils.execute(
                        str(pulpGccSrcDir / "configure"),
                        *pulpGccArgs,
                        cwd=pulpGccBuildDir,
                        env=env,
                        live=verbose,
                    )
                utils.make(
                    cwd=pulpGccBuildDir,
                    env=env,
                    live=verbose,
                    threads=threads,
                )
            else:
                pulpGccUrl = user_vars.get("pulp_gcc.dl_url", None)
                assert pulpGccUrl, "pulp_gcc.dl_url undefined and environment and pulp_gcc.from_source=0"
                pulpGccBaseUrl, pulpGccArchive = pulpGccUrl.rsplit("/", 1)
                utils.download_and_extract(pulpGccBaseUrl, pulpGccArchive, pulpGccInstallDir)
    if "pulp_gcc.name" in user_vars:
        pulpGccName = user_vars["pulp_gcc.name"]
    else:
        pulpGccName = "riscv32-unknown-elf"
    context.cache["pulp_gcc.install_dir", flags] = pulpGccInstallDir
    context.cache["pulp_gcc.name", flags] = pulpGccName
    # context.cache["pulp_gcc.build_dir", flags] = pulpGccBuildDir


@Tasks.provides(["pulp_freertos.src_dir", "pulp_freertos.support_dir", "pulp_freertos.config_dir"])
@Tasks.validate(_validate_pulp)
@Tasks.register(category=TaskType.TARGET)
def clone_pulp_freertos(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the pulp-freertos repository."""
    pulpRtosName = utils.makeDirName("pulp_freertos")
    pulpRtosSrcDir = context.environment.paths["deps"].path / "src" / pulpRtosName
    pulpRtosSupportDir = pulpRtosSrcDir / "support"
    pulpConfigsDir = pulpRtosSupportDir / "pulp-configs" / "configs"
    if (
        rebuild
        or not utils.is_populated(pulpRtosSrcDir)
        or not utils.is_populated(pulpRtosSupportDir)
        or not utils.is_populated(pulpConfigsDir)
    ):
        pulpRtosRepo = context.environment.repos["pulp_freertos"]
        utils.clone_wrapper(pulpRtosRepo, pulpRtosSrcDir, refresh=rebuild)
        user_vars = context.environment.vars
        experimental_install = user_vars.get("pulp_freertos.experimental_install", False)
        if experimental_install:
            patchFile = Path(
                pkg_resources.resource_filename(
                    "mlonmcu", os.path.join("..", "resources", "patches", "pulp_freertos_support.patch")
                )
            )
            if patchFile.is_file():
                utils.patch(patchFile, cwd=pulpRtosSrcDir)
    context.cache["pulp_freertos.src_dir"] = pulpRtosSrcDir
    context.cache["pulp_freertos.support_dir"] = pulpRtosSupportDir
    context.cache["pulp_freertos.config_dir"] = pulpConfigsDir


@Tasks.needs(["pulp_freertos.src_dir", "pulp_freertos.support_dir", "pulp_freertos.config_dir"])
@Tasks.provides(["pulp_freertos.install_dir", "pulp_freertos.pythonpath", "gvsoc.exe"])
@Tasks.validate(_validate_pulp)
@Tasks.register(category=TaskType.TARGET)
def install_gvsoc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Setup the pulp-freertos build."""
    pulpRtosName = utils.makeDirName("pulp_freertos")
    user_vars = context.environment.vars
    experimental_install = user_vars.get("pulp_freertos.experimental_install", False)
    pulpRtosSupportDir = context.cache["pulp_freertos.support_dir"]
    gvsocExe = pulpRtosSupportDir / "egvsoc.sh"
    pulpConfigDir = context.cache["pulp_freertos.config_dir"]
    if experimental_install:
        pulpRtosInstallDir = context.environment.paths["deps"].path / "install" / pulpRtosName
    else:
        pulpRtosInstallDir = pulpRtosSupportDir / "install"
    pulpPythonPath = pulpRtosInstallDir / "python"
    if rebuild or not utils.is_populated(pulpRtosInstallDir) or not utils.is_populated(pulpPythonPath):
        utils.mkdirs(pulpRtosInstallDir)
        env = os.environ.copy()
        env.update(
            {
                "PULP_CURRENT_CONFIG": "pulpissimo@config_file=chips/pulpissimo/pulpissimo.json",
                "PULP_CONFIGS_PATH": pulpConfigDir,
                "PYTHONPATH": pulpRtosInstallDir / "python",
                "INSTALL_DIR": pulpRtosInstallDir,
                "ARCHI_DIR": pulpRtosSupportDir / "archi" / "include",
                "SUPPORT_ROOT": pulpRtosSupportDir,
            }
        )
        if experimental_install:
            env.update(
                {
                    "SUPPORT_INSTALL_DIR": pulpRtosInstallDir,
                }
            )
        makeArgs = ["-f", pulpRtosSupportDir / "support.mk", "gvsoc"]
        utils.make(
            *makeArgs,
            cwd=pulpRtosInstallDir,
            env=env,
            live=verbose,
            threads=1,  # Script returns 2 exit code if parallel
        )
    context.cache["pulp_freertos.install_dir"] = pulpRtosInstallDir
    context.cache["pulp_freertos.pythonpath"] = pulpPythonPath
    context.cache["gvsoc.exe"] = gvsocExe
    context.export_paths.add(gvsocExe.parent)
