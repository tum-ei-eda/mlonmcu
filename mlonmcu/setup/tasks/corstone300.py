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

from .common import _validate_gcc, get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_corstone300(context: MlonMcuContext, params=None):
    return context.environment.has_target("corstone300") and _validate_gcc(context, params=params)


@Tasks.provides(["corstone300.exe"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_corstone300(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install corstone300 FVP from ARM."""
    fvpName = utils.makeDirName("corstone300")
    fvpInstallDir = context.environment.paths["deps"].path / "install" / fvpName
    fvpSubDir = fvpInstallDir / "fvp"
    fvpExe = fvpSubDir / "models" / "Linux64_GCC-6.4" / "FVP_Corstone_SSE-300_Ethos-U55"
    user_vars = context.environment.vars
    if "corstone300.exe" in user_vars:  # TODO: also check command line flags?
        # fvpExe = user_vars["corstone300.exe"]
        return False
    else:
        if not fvpExe.is_file():
            fvpUrl = "https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/"
            fvpFileName = "FVP_Corstone_SSE-300_11.16_26"
            fvpArchive = fvpFileName + ".tgz"
            utils.download_and_extract(fvpUrl, fvpArchive, fvpInstallDir, progress=verbose)
            fvpScript = fvpInstallDir / "FVP_Corstone_SSE-300.sh"
            utils.execute(
                fvpScript,
                "--i-agree-to-the-contained-eula",
                "--no-interactive",
                "-d",
                fvpSubDir,
            )
    context.cache["corstone300.exe"] = fvpExe
    context.export_paths.add(fvpExe.parent)
