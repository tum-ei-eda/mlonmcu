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


def _validate_tflite_visualize(context: MlonMcuContext, params=None):
    return context.environment.has_frontend("tflite") and context.environment.has_feature("visualize")


@Tasks.provides(["tflite_visualize.exe"])
@Tasks.validate(_validate_tflite_visualize)
@Tasks.register(category=TaskType.FEATURE)
def download_tflite_vizualize(
    context: MlonMcuContext, params=None, rebuild=False, verbose=False, threads=multiprocessing.cpu_count()
):
    """Download the visualize.py script for TFLite."""
    # This script is content of the tensorflow repo (not tflite-micro) and unfortunately not bundled
    # into the tensorflow python package. Therefore just download this single file form GitHub

    tfLiteVizualizeName = utils.makeDirName("tflite_visualize")
    tfLiteVizualizeInstallDir = context.environment.paths["deps"].path / "install" / tfLiteVizualizeName
    tfLiteVizualizeExe = tfLiteVizualizeInstallDir / "visualize.py"
    user_vars = context.environment.vars
    if "tflite_visualize.exe" in user_vars:  # TODO: also check command line flags?
        return False
    if rebuild or not utils.is_populated(tfLiteVizualizeInstallDir):
        tfLiteVizualizeInstallDir.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/visualize.py"
        utils.download(url, tfLiteVizualizeExe, progress=verbose)
    context.cache["tflite_visualize.exe"] = tfLiteVizualizeExe
    context.export_paths.add(tfLiteVizualizeInstallDir)
