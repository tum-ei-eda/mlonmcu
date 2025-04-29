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

# from mlonmcu.setup import utils
# from mlonmcu.timeout import exec_timeout
# from mlonmcu.config import str2bool, str2list, str2dict
from mlonmcu.logging import get_logger

# from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat
from .backend import IREEBackend

logger = get_logger()


class IREEVMVXBackend(IREEBackend):
    registry = {}

    name = None

    FEATURES = IREEBackend.FEATURES | set()

    DEFAULTS = IREEBackend.DEFAULTS

    REQUIRED = IREEBackend.REQUIRED | set()

    OPTIONAL = IREEBackend.OPTIONAL

    name = "ireevmvx"

    def __init__(self, features=None, config=None):
        super().__init__(
            output_format="vm-bytecode", hal_backend="vmvx", hal_inline=False, features=features, config=config
        )
