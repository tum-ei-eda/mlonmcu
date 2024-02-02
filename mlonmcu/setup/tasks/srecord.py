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


def _validate_srecord(context: MlonMcuContext, params={}):
    return context.environment.has_target("cv32e40p") or context.environment.has_target("vicuna")


# TODO: cleanup build dir?


@Tasks.provides(["srecord.src_dir"])
@Tasks.validate(_validate_srecord)
@Tasks.register(category=TaskType.MISC)
def clone_srecord(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the screcord repository."""
    name = utils.makeDirName("srecord")
    user_vars = context.environment.vars
    if "srecord.src_dir" in user_vars:
        srcDir = Path(user_vars["srecord.src_dir"])
        rebuild = False
    else:
        srcDir = context.environment.paths["deps"].path / "src" / name
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["srecord"]
        utils.clone_wrapper(repo, srcDir, refresh=rebuild)
    context.cache["srecord.src_dir"] = srcDir


# Only build srec_cat here
# @Tasks.needs(["etiss.src_dir", "llvm.install_dir"])
@Tasks.needs(["srecord.src_dir"])
@Tasks.provides(["srecord.build_dir", "srecord.install_dir"])
@Tasks.validate(_validate_srecord)
@Tasks.register(category=TaskType.MISC)
def build_srecord(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build the srecord unitilties."""
    name = utils.makeDirName("srecord")
    buildDir = context.environment.paths["deps"].path / "build" / name
    installDir = context.environment.paths["deps"].path / "install" / name
    user_vars = context.environment.vars
    if "srecord.install_dir" in user_vars:
        return False
    utils.mkdirs(installDir)
    if rebuild or not utils.is_populated(buildDir):
        utils.mkdirs(buildDir)
        utils.cmake(
            context.cache["srecord.src_dir"],
            "-DCMAKE_INSTALL_PREFIX=" + str(installDir),
            cwd=buildDir,
            debug=False,
            live=verbose,
        )
        utils.make("srec_cat", cwd=buildDir, threads=threads, live=verbose)
        utils.copy(buildDir / "srec_cat" / "srec_cat", installDir / "srec_cat")
    context.cache["srecord.build_dir"] = buildDir
    context.cache["srecord.install_dir"] = installDir
