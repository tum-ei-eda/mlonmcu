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
import stat
import multiprocessing

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_corev_ovpsim(context: MlonMcuContext, params=None):
    return context.environment.has_target("corev_ovpsim")


@Tasks.provides(["corev_ovpsim.exe", "corev_ovpsim.install_dir"])
@Tasks.validate(_validate_corev_ovpsim)
@Tasks.register(category=TaskType.TARGET)
def install_corev_ovpsim(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install CORE-V OVPSim."""
    ovpName = utils.makeDirName("corev_ovpsim")
    ovpInstallDir = context.environment.paths["deps"].path / "install" / ovpName
    # TODO: windows?
    ovpExe = ovpInstallDir / "bin" / "Linux64" / "riscvOVPsimCOREV.exe"
    user_vars = context.environment.vars
    if "corev_ovpsim.exe" in user_vars:  # TODO: also check command line flags?
        return False
    else:
        if not ovpExe.is_file():

            def _helper(url):
                fullUrlSplit = url.split("/")
                tmpUrl = "/".join(fullUrlSplit[:-1])
                tmpFileName, tmpFileExtension = fullUrlSplit[-1].split(".", 1)
                return tmpUrl, tmpFileName, tmpFileExtension

            if "corev_ovpsim.dl_url" in user_vars:
                ovpUrl, ovpFileName, ovpFileExtension = _helper(user_vars["corev_ovpsim.dl_url"])
            else:
                ovpUrl = "https://github.com/openhwgroup/riscv-ovpsim-corev/archive/refs/heads/"
                # ovpVersion = user_vars.get("corev_ovpsim.version", "v20230425")
                ovpVersion = user_vars.get("corev_ovpsim.version", "v20230724")
                ovpFileName = str(ovpVersion)
                ovpFileExtension = "zip"
            ovpArchive = ovpFileName + "." + ovpFileExtension
            utils.download_and_extract(ovpUrl, ovpArchive, ovpInstallDir, progress=verbose)
            st = os.stat(ovpExe)
            # make executable
            os.chmod(ovpExe, st.st_mode | stat.S_IEXEC)
    context.cache["corev_ovpsim.install_dir"] = ovpInstallDir
    context.cache["corev_ovpsim.exe"] = ovpExe
