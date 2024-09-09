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

########
# ara  #
########


def _validate_ara(context: MlonMcuContext, params=None):
    return context.environment.has_target("ara")


def _validate_ara_rtl(context: MlonMcuContext, params=None):
    return context.environment.has_target("ara_rtl")


@Tasks.provides(["ara.src_dir"])
@Tasks.validate(_validate_ara)
@Tasks.register(category=TaskType.TARGET)
def clone_ara(context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()):
    """Clone the ara repository."""
    if not params:
        params = {}
    user_vars = context.environment.vars
    flags = utils.makeFlags()
    araName = utils.makeDirName("ara", flags=flags)
    araSrcDir = context.environment.paths["deps"].path / "src" / araName
    if "ara.src_dir" in user_vars:  # TODO: also check command line flags?
        # This would overwrite the cache.ini entry which is NOT wanted! -> return false but populate gcc_name?
        araSrcDir = user_vars["ara.src_dir"]
    else:
        if rebuild or not utils.is_populated(araSrcDir):
            araRepo = context.environment.repos["ara"]
            utils.clone_wrapper(araRepo, araSrcDir, refresh=rebuild)
            utils.execute("make", "apply-patches", cwd=araSrcDir / "hardware")
            utils.execute("make", "bender", cwd=araSrcDir / "hardware")
    context.cache["ara.src_dir", flags] = araSrcDir
