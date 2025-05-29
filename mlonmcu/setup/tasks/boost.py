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


def _validate_boost(context: MlonMcuContext, params=None):
    user_vars = context.environment.vars
    use_system_boost = user_vars.get("boost.use_system", False)
    if use_system_boost:
        return False
    if context.environment.has_target("spike") or context.environment.has_target("etiss_pulpino"):
        return True
    return False


@Tasks.provides(["boost.src_dir"])
@Tasks.validate(_validate_boost)
@Tasks.register(category=TaskType.MISC)
def download_boost(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Fetch the boost library sources."""
    user_vars = context.environment.vars
    version = user_vars.get("boost.version", "1.81.0")
    # flags = utils.makeFlags((True, version))
    flags = []
    boostName = utils.makeDirName("boost", flags=flags)
    boostSrcDir = context.environment.paths["deps"].path / "src" / boostName
    if "boost.install_dir" in user_vars:
        return False
    if rebuild or not utils.is_populated(boostSrcDir):
        if "boost.dl_url" in user_vars:
            boostUrl = user_vars["boost.dl_url"]
            boostUrl, boostArchive = boostUrl.rsplit("/", 1)
        else:
            # boostUrl = f"https://boostorg.jfrog.io/artifactory/main/release/{version}/source/"
            boostUrl = f"https://archives.boost.io/release/{version}/source/"
            version_ = version.replace(".", "_")
            boostArchive = f"boost_{version_}.tar.gz"  # zip does not preserve file permissions
        utils.download_and_extract(boostUrl, boostArchive, boostSrcDir, progress=verbose)
    context.cache["boost.src_dir"] = boostSrcDir


@Tasks.needs(["boost.src_dir"])
@Tasks.provides(["boost.install_dir"])
@Tasks.validate(_validate_boost)
@Tasks.register(category=TaskType.MISC)
def build_boost(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Build boost library."""
    flags = []
    boostName = utils.makeDirName("boost", flags=flags)
    boostSrcDir = context.cache["boost.src_dir"]
    # boostBuildDir = context.environment.paths["deps"].path / "build" / boostName
    boostBuildDir = boostSrcDir
    boostInstallDir = context.environment.paths["deps"].path / "install" / boostName
    user_vars = context.environment.vars
    if "boost.install_dir" in user_vars:
        return False
    if rebuild or not utils.is_populated(boostInstallDir):
        bootstrapArgs = [
            "--with-libraries=log,thread,system,filesystem,program_options,test",
            f"--prefix={boostInstallDir}",
        ]
        # env = os.environ.copy()
        utils.mkdirs(boostBuildDir)
        utils.execute(
            "sh",
            str(boostSrcDir / "bootstrap.sh"),
            *bootstrapArgs,
            cwd=boostBuildDir,
            # env=env,
            live=False,
            # print_output=False,
        )
        utils.execute(
            str(boostBuildDir / "b2"),
            "install",
            cwd=boostBuildDir,
            # env=env,
            live=False,
            # print_output=False,
        )
    context.cache["boost.install_dir"] = boostInstallDir
