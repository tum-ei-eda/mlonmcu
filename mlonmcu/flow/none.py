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
"""Dummy framework and backend."""

# from mlonmcu.flow.none.backend.none import NoneBackend
from .framework import Framework
from .backend import Backend


class NoneFramework(Framework):
    """TODO."""

    name = "none"

    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(features=features, config=config)


class NoneBackend(Backend):
    registry = {}

    name = None

    FEATURES = set()

    DEFAULTS = {}

    OPTIONAL = set()

    REQUIRED = set()

    name = "none"

    def __init__(self, features=None, config=None):
        super().__init__(framework="none", features=features, config=config)

    def generate(self):
        return None

    def load_model(
        self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None, params_path=None
    ):
        pass
