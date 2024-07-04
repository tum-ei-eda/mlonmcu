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


def _validate_mlif(context: MlonMcuContext, params=None):
    return context.environment.has_platform("mlif")


@Tasks.provides(["mlif.src_dir"])
@Tasks.validate(_validate_mlif)
@Tasks.register(category=TaskType.PLATFORM)
def clone_mlif(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the MLonMCU SW repository."""
    mlifName = utils.makeDirName("mlif")
    mlifSrcDir = context.environment.paths["deps"].path / "src" / mlifName
    if rebuild or not utils.is_populated(mlifSrcDir):
        mlifRepo = context.environment.repos["mlif"]
        utils.clone_wrapper(mlifRepo, mlifSrcDir, refresh=rebuild)
    context.cache["mlif.src_dir"] = mlifSrcDir
