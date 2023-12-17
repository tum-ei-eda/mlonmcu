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


def _validate_benchmarks(context: MlonMcuContext, params=None):
    if params:
        benchmark = params.get("benchmark", None)
        assert benchmark is not None
        if not context.environment.has_frontend(benchmark):
            return False
        user_vars = context.environment.vars
        if f"{benchmark}.src_dir" in user_vars:
            return False
    return True


# Cloning all benchmarks in a single task saves a lot of code
@Tasks.provides(["benchmarks.src_dir"])
@Tasks.param("benchmark", ["embench", "taclebench", "polybench", "mibench"])
@Tasks.validate(_validate_benchmarks)
@Tasks.register(category=TaskType.FRONTEND)
def clone_benchmarks(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the embench repo."""
    benchmark = params.get("benchmark", None)
    benchName = utils.makeDirName(benchmark)
    benchmarksSrcDir = context.environment.paths["deps"].path / "src" / "benchmarks"
    benchSrcDir = benchmarksSrcDir / benchName
    utils.mkdirs(benchmarksSrcDir)
    if rebuild or not utils.is_populated(benchSrcDir):
        benchRepo = context.environment.repos[benchmark]
        utils.clone_wrapper(benchRepo, benchSrcDir, refresh=rebuild)
    context.cache["benchmarks.src_dir"] = benchmarksSrcDir
    context.cache[f"{benchmark}.src_dir"] = benchSrcDir
