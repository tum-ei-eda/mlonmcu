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


def _validate_tensorflow(context: MlonMcuContext, params=None):
    return context.environment.has_framework("tflm")


@Tasks.provides(["tf.src_dir"])
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def clone_tensorflow(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Clone the TF/TFLM repository."""
    tfName = utils.makeDirName("tf")
    tfSrcDir = context.environment.paths["deps"].path / "src" / tfName
    if rebuild or not utils.is_populated(tfSrcDir):
        tfRepo = context.environment.repos["tensorflow"]
        utils.clone(tfRepo.url, tfSrcDir, branch=tfRepo.ref, refresh=rebuild)
    context.cache["tf.src_dir"] = tfSrcDir


@Tasks.needs(["tf.src_dir"])
@Tasks.provides(["tf.dl_dir", "tf.lib_path"])
# @Tasks.param("dbg", False)
@Tasks.param("dbg", True)
@Tasks.validate(_validate_tensorflow)
@Tasks.register(category=TaskType.FRAMEWORK)
def build_tensorflow(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download tensorflow dependencies and build lib."""
    if not params:
        params = {}
    flags = utils.makeFlags((params["dbg"], "dbg"))
    # tfName = utils.makeDirName("tf", flags=flags)
    tfSrcDir = context.cache["tf.src_dir"]
    tflmDir = Path(tfSrcDir) / "tensorflow" / "lite" / "micro"
    tflmBuildDir = tflmDir / "tools" / "make"
    tflmDownloadsDir = tflmBuildDir / "downloads"
    if params["dbg"]:
        tflmLib = (
            tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
        )  # FIXME: add _dbg suffix is possible
    else:
        tflmLib = tflmBuildDir / "gen" / "linux_x86_64" / "lib" / "libtensorflow-microlite.a"
    # if rebuild or not tflmLib.is_file() or not utils.is_populated(tflmDownloadsDir):
    if rebuild or not utils.is_populated(tflmDownloadsDir):
        tfDbgArg = ["BUILD_TYPE=debug"] if params["dbg"] else []
        utils.make(
            "-f",
            str(tflmDir / "tools" / "make" / "Makefile"),
            "third_party_downloads",
            threads=threads,
            *tfDbgArg,
            cwd=tfSrcDir,
            live=verbose,
        )
    context.cache["tf.dl_dir"] = tflmDownloadsDir
    context.cache["tf.lib_path", flags] = tflmLib  # ignore!
