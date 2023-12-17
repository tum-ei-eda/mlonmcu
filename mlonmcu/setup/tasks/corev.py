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
from .cv32e40p import _validate_cv32e40p

logger = get_logger()

Tasks = get_task_factory()


def _validate_corevverif(context: MlonMcuContext, params=None):
    if _validate_cv32e40p(context, params=params):
        return True
    return False


@Tasks.provides(["corevverif.src_dir"])
@Tasks.validate(_validate_corevverif)
@Tasks.register(category=TaskType.TARGET)
def clone_corevverif(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the core-v-verif repo."""
    name = utils.makeDirName("corevverif")
    srcDir = context.environment.paths["deps"].path / "src" / name
    user_vars = context.environment.vars
    if "corevverif.src_dir" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["corevverif"]
        utils.clone_wrapper(repo, srcDir, refresh=rebuild)
    context.cache["corevverif.src_dir"] = srcDir
