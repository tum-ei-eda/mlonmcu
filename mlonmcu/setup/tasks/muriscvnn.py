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
from pathlib import Path

from mlonmcu.setup.task import TaskType
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.setup import utils
from mlonmcu.logging import get_logger

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_muriscvnn(context: MlonMcuContext, params=None):
    if not context.environment.supports_feature("muriscvnn"):
        return False
    user_vars = context.environment.vars
    if "muriscvnn.src_dir" not in user_vars:
        assert "muriscvnn" in context.environment.repos, "Undefined repository: 'muriscvnn'"
    if params:
        toolchain = params.get("toolchain", "gcc")
        if not context.environment.has_toolchain(toolchain):
            return False
        target_arch = params.get("target_arch", "riscv")
        if target_arch == "riscv":
            if params.get("vext", False):
                if not context.environment.supports_feature("vext"):
                    return False
            if params.get("pext", False):
                if toolchain == "llvm":
                    # Unsupported
                    return False
                if not context.environment.supports_feature("pext"):
                    return False
            if params.get("vext", False) and params.get("pext", False):
                # Either pext or vext!
                return False
        elif target_arch == "x86":
            if toolchain != "gcc":
                return False
            if params.get("vext", False) or params.get("pext", False):
                return False
            # TODO: validate chosen toolchain?
    return True


@Tasks.provides(["muriscvnn.src_dir", "muriscvnn.inc_dir"])
@Tasks.validate(_validate_muriscvnn)
@Tasks.register(category=TaskType.OPT)
def clone_muriscvnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the muRISCV-NN project."""
    muriscvnnName = utils.makeDirName("muriscvnn")
    user_vars = context.environment.vars
    # if "muriscvnn.lib" in user_vars:  # TODO: also check command line flags?
    #     return False
    if "muriscvnn.src_dir" in user_vars:
        muriscvnnSrcDir = user_vars["muriscvnn.src_dir"]
        muriscvnnIncludeDir = Path(muriscvnnSrcDir) / "Include"
        rebuild = False
    else:
        muriscvnnSrcDir = context.environment.paths["deps"].path / "src" / muriscvnnName
        muriscvnnIncludeDir = muriscvnnSrcDir / "Include"
    if rebuild or not utils.is_populated(muriscvnnSrcDir):
        muriscvnnRepo = context.environment.repos["muriscvnn"]
        utils.clone_wrapper(muriscvnnRepo, muriscvnnSrcDir, refresh=rebuild)
    context.cache["muriscvnn.src_dir"] = muriscvnnSrcDir
    context.cache["muriscvnn.inc_dir"] = muriscvnnIncludeDir
