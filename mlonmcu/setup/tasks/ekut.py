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

##############
# hannah-tvm #
##############


def _validate_hannah_tvm(context: MlonMcuContext, params=None):
    return context.environment.has_target("microtvm_gvsoc")


@Tasks.provides(["hannah_tvm.src_dir", "microtvm_gvsoc.template"])
@Tasks.validate(_validate_hannah_tvm)
@Tasks.register(category=TaskType.TARGET)
def clone_hannah_tvm(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the hannah-tvm repository."""
    hannahTvmName = utils.makeDirName("hannah_tvm")
    hannahTvmSrcDir = context.environment.paths["deps"].path / "src" / hannahTvmName
    if rebuild or not utils.is_populated(hannahTvmSrcDir):
        pulpRtosRepo = context.environment.repos["hannah_tvm"]
        utils.clone(pulpRtosRepo.url, hannahTvmSrcDir, branch=pulpRtosRepo.ref, refresh=rebuild, recursive=True)
    context.cache["hannah_tvm.src_dir"] = hannahTvmSrcDir
    context.cache["microtvm_gvsoc.template"] = hannahTvmSrcDir / "template" / "gvsoc"
