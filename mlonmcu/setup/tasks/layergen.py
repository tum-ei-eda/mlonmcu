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


def _validate_layergen(context: MlonMcuContext, params=None):
    return context.environment.has_frontend("layergen")


@Tasks.provides(["layergen.src_dir", "layergen.exe"])
@Tasks.validate(_validate_layergen)
@Tasks.register(category=TaskType.FEATURE)
def clone_layergen(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the repo with the layergen scripts."""
    name = utils.makeDirName("layergen")
    srcDir = context.environment.paths["deps"].path / "src" / name
    if rebuild or not utils.is_populated(srcDir):
        repo = context.environment.repos["layergen"]
        utils.clone(repo.url, srcDir, branch=repo.ref, refresh=rebuild)
    context.cache["layergen.src_dir"] = srcDir
    context.cache["layergen.exe"] = srcDir / "gen_model.py"
