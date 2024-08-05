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
import venv
import multiprocessing
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_zephyr(context: MlonMcuContext, params=None):
    return context.environment.has_platform("zephyr")


# @Tasks.needs([])
@Tasks.provides(["zephyr.install_dir", "zephyr.sdk_dir", "zephyr.venv_dir"])
@Tasks.validate(_validate_zephyr)
@Tasks.register(category=TaskType.PLATFORM)
def install_zephyr(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install support for the Zephyr Platform."""
    zephyrName = utils.makeDirName("zephyr")
    zephyrInstallDir = context.environment.paths["deps"].path / "install" / zephyrName
    zephyrInstallDir.mkdir(exist_ok=True)
    zephyrSdkDir = zephyrInstallDir / "sdk"
    zephyrVenvDir = zephyrInstallDir / "venv"
    zephyrModulesDir = zephyrInstallDir / "modules"
    user_vars = context.environment.vars
    if "zephyr.install_dir" in user_vars:  # TODO: also check command line flags?
        assert "zephyr.sdk_dir" in user_vars
        assert "zephyr.venv_dir" in user_vars
        return False
    # boards = ["espressif"]
    # if "zephyr.boards" in user_vars:
    #     boards = user_vars["zephyr.boards"]
    # if not isinstance(boards, str):
    #     assert isinstance(boards, list)
    #     boards = ",".join(boards)
    if (
        not utils.is_populated(zephyrInstallDir)
        or not utils.is_populated(zephyrSdkDir)
        or not utils.is_populated(zephyrVenvDir)
        or not utils.is_populated(zephyrModulesDir)
        or rebuild
    ):
        zephyrVenvScript = zephyrVenvDir / "bin" / "activate"
        if not utils.is_populated(zephyrVenvDir):
            venv.create(zephyrVenvDir)
        utils.execute(f". {zephyrVenvScript} && pip install west", shell=True, live=verbose)
        zephyrRepo = context.environment.repos["zephyr"]
        zephyrUrl = zephyrRepo.url
        if not utils.is_populated(zephyrInstallDir / "zephyr"):
            zephyrVersion = zephyrRepo.ref
            utils.execute(
                f". {zephyrVenvScript} && west init -m {zephyrUrl} --mr {zephyrVersion} {zephyrInstallDir}",
                shell=True,
                live=verbose,
            )
        extra = zephyrInstallDir / "zephyr" / "scripts" / "requirements.txt"
        utils.execute(f". {zephyrVenvScript} && pip install -r {extra}", shell=True, live=verbose)
        env = os.environ.copy()
        env["ZEPHYR_BASE"] = str(zephyrInstallDir / "zephyr")
        utils.execute(f". {zephyrVenvScript} && west update", shell=True, live=verbose, env=env)
        if "zephyr.sdk_version" in user_vars:
            sdkVersion = user_vars["zephyr.sdk_version"]
        else:
            sdkVersion = "0.15.0-rc1"
        sdkDist = "linux-x86_64"
        sdkUrl = f"https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v{sdkVersion}/"
        sdkArchive = f"zephyr-sdk-{sdkVersion}_{sdkDist}.tar.gz"
        if not utils.is_populated(zephyrSdkDir):
            utils.download_and_extract(sdkUrl, sdkArchive, zephyrSdkDir, progress=verbose)
        sdkScript = zephyrSdkDir / "setup.sh"
        # TODO: allow to limit installed toolchains
        utils.execute(sdkScript, "-t", "all", "-h", live=verbose)
        # Apply patch to fix esp32c3 support
        patchFile = Path(
            pkg_resources.resource_filename(
                "mlonmcu", os.path.join("..", "resources", "patches", "zephyr", "fix_esp32c3_march.patch")
            )
        )
        if patchFile.is_file():
            xtensaDir = zephyrInstallDir / "modules" / "hal" / "xtensa"
            utils.patch(patchFile, cwd=xtensaDir)
    context.cache["zephyr.install_dir"] = zephyrInstallDir
    context.cache["zephyr.sdk_dir"] = zephyrSdkDir
    context.cache["zephyr.venv_dir"] = zephyrVenvDir
