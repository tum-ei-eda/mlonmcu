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

from .common import get_task_factory

logger = get_logger()

Tasks = get_task_factory()


def _validate_dtc(context: MlonMcuContext, params=None):
    # TODO: require this for: ???
    # TODO: cleanup after build
    # return True
    return False


@Tasks.provides(["dtc.src_dir"])
@Tasks.validate(_validate_dtc)
@Tasks.register(category=TaskType.MISC)
def clone_dtc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the dtc repo."""
    name = utils.makeDirName("dtc")
    srcDir = context.environment.paths["deps"].path / "src" / name
    user_vars = context.environment.vars
    if "dtc.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["dtc"]
        utils.clone(repo.url, srcDir, branch=repo.ref)
    context.cache["dtc.src_dir"] = srcDir


@Tasks.needs(["dtc.src_dir"])
@Tasks.provides(["dtc.install_dir", "dtc.build_dir", "dtc.exe"])
@Tasks.validate(_validate_dtc)
@Tasks.register(category=TaskType.MISC)
def build_dtc(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Build the device tree compile."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    if "dtc.exe" in user_vars:  # TODO: also check command line flags?
        return False
    name = utils.makeDirName("dtc")
    srcDir = context.cache["dtc.src_dir"]
    buildDir = context.environment.paths["deps"].path / "build" / name
    installDir = context.environment.paths["deps"].path / "install" / name
    exe = installDir / "bin" / "dtc"
    if rebuild or not (utils.is_populated(buildDir) and exe.is_file()):
        utils.mkdirs(buildDir)
        # utils.make("install", "-C", srcDir, f"PREFIX={installDir}", cwd=buildDir, threads=threads, live=verbose)
        utils.make("dtc", "-C", srcDir, f"PREFIX={installDir}", cwd=buildDir, threads=threads, live=verbose)
        utils.mkdirs(installDir / "bin")
        utils.copy(srcDir / "dtc", installDir / "bin" / "dtc")
    context.cache["dtc.build_dir"] = buildDir
    context.cache["dtc.install_dir"] = installDir
    context.cache["dtc.exe"] = exe
    context.export_paths.add(installDir)
