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


def _validate_cfu_playground(context: MlonMcuContext, params=None):
    return context.environment.has_platform("cfu_playground")


@Tasks.provides(["cfu_playground.src_dir"])
@Tasks.validate(_validate_cfu_playground)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_cfu_playground(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the CFU Playground repository."""
    cfuName = utils.makeDirName("cfu_playground")
    cfuSrcDir = context.environment.paths["deps"].path / "src" / cfuName
    if rebuild or not utils.is_populated(cfuSrcDir):
        cfuRepo = context.environment.repos["cfu_playground"]
        utils.clone_wrapper(cfuRepo, cfuSrcDir, refresh=rebuild)
    context.cache["cfu_playground.src_dir"] = cfuSrcDir


@Tasks.needs(["cfu_playground.src_dir"])
# @Tasks.provides(["tf.dl_dir", "tf.lib_path"])
@Tasks.validate(_validate_cfu_playground)
@Tasks.register(category=TaskType.FRAMEWORK)
def setup_cfu_playground(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Setup CFU Playground."""
    if not params:
        params = {}
    # cfuName = utils.makeDirName("cfu_playground", flags=flags)
    cfuSrcDir = context.cache["cfu_playground.src_dir"]
    check_paths = [
        cfuSrcDir / "third_party" / "python" / "pythondata-software-picolibc",
        cfuSrcDir / "third_party" / "renode" / "plugins" / "VerilatorIntegrationLibrary" / "src" / "renode_cfu.h",
    ]
    if rebuild or not all(utils.is_populated(path) for path in check_paths):
        setupScript = cfuSrcDir / "scripts" / "setup"
        utils.execute(
            setupScript,
            # env=env,
            live=verbose,
            # print_output=False,
            cwd=cfuSrcDir,
        )
    # context.cache["cfu_playground.dl_dir"] = tflmDownloadsDir
