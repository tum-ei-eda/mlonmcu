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


def _validate_cmsisnn(context: MlonMcuContext, params=None):
    if not (context.environment.has_feature("cmsisnn") or context.environment.has_feature("cmsisnnbyoc")):
        return False
    mvei = params.get("mvei", False)
    dsp = params.get("dsp", False)
    target_arch = params.get("target_arch", None)
    if target_arch == "arm":
        if dsp and not context.environment.has_feature("arm_dsp"):
            return False
        if mvei and not context.environment.has_feature("arm_mvei"):
            return False
    else:
        if mvei or dsp:
            return False
    return True


@Tasks.provides(["cmsisnn.dir"])
@Tasks.validate(_validate_cmsisnn)
@Tasks.register(category=TaskType.OPT)
def clone_cmsisnn(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """CMSIS-NN repository."""
    cmsisName = utils.makeDirName("cmsisnn")
    cmsisnnSrcDir = Path(context.environment.paths["deps"].path) / "src" / cmsisName
    if rebuild or not utils.is_populated(cmsisnnSrcDir):
        cmsisnnRepo = context.environment.repos["cmsisnn"]
        utils.clone_wrapper(cmsisnnRepo, cmsisnnSrcDir, refresh=rebuild)
    context.cache["cmsisnn.dir"] = cmsisnnSrcDir


def _validate_cmsis(context: MlonMcuContext, params=None):
    return _validate_cmsisnn(context, params=params) or context.environment.has_target("corstone300")


@Tasks.provides(["cmsis.dir"])
@Tasks.validate(_validate_cmsis)
@Tasks.register(category=TaskType.MISC)
def clone_cmsis(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """CMSIS repository."""
    cmsisName = utils.makeDirName("cmsis")
    cmsisSrcDir = Path(context.environment.paths["deps"].path) / "src" / cmsisName
    if rebuild or not utils.is_populated(cmsisSrcDir):
        cmsisRepo = context.environment.repos["cmsis"]
        utils.clone_wrapper(cmsisRepo, cmsisSrcDir, refresh=rebuild)
    context.cache["cmsis.dir"] = cmsisSrcDir


def _validate_ethosu_platform(context: MlonMcuContext, params=None):
    return context.environment.has_target("corstone300")


@Tasks.provides(["ethosu_platform.dir"])
@Tasks.validate(_validate_ethosu_platform)
@Tasks.register(category=TaskType.MISC)
def clone_ethosu_platform(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """EthosU platform repository."""
    ethosuPlatformName = utils.makeDirName("ethosu_platform")
    ethosuPlatformSrcDir = Path(context.environment.paths["deps"].path) / "src" / ethosuPlatformName
    if rebuild or not utils.is_populated(ethosuPlatformSrcDir):
        ethosuPlatformRepo = context.environment.repos["ethosu_platform"]
        utils.clone_wrapper(ethosuPlatformRepo, ethosuPlatformSrcDir, refresh=rebuild)
    context.cache["ethosu_platform.dir"] = ethosuPlatformSrcDir
