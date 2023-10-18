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
from .corstone300 import _validate_corstone300

logger = get_logger()
Tasks = get_task_factory()


def _validate_arm_gcc(context: MlonMcuContext, params=None):
    return _validate_corstone300(context, params=params) and _validate_gcc(context, params=params)


@Tasks.provides(["arm_gcc.install_dir"])
@Tasks.validate(_validate_corstone300)
@Tasks.register(category=TaskType.TARGET)
def install_arm_gcc(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download and install GNU compiler toolchain from ARM."""
    armName = utils.makeDirName("arm_gcc")
    armInstallDir = context.environment.paths["deps"].path / "install" / armName
    user_vars = context.environment.vars
    if "arm_gcc.install_dir" in user_vars:  # TODO: also check command line flags?
        # armInstallDir = user_vars["riscv_gcc.install_dir"]
        return False
    else:
        if not utils.is_populated(armInstallDir):
            armUrl = "https://developer.arm.com/-/media/Files/downloads/gnu/11.2-2022.02/binrel/"
            armFileName = "gcc-arm-11.2-2022.02-x86_64-arm-none-eabi"
            armArchive = armFileName + ".tar.xz"
            utils.download_and_extract(armUrl, armArchive, armInstallDir, progress=verbose)
    context.cache["arm_gcc.install_dir"] = armInstallDir
    context.export_paths.add(armInstallDir / "bin")
