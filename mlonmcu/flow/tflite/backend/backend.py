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
from mlonmcu.flow.backend import Backend


class TFLiteBackend(Backend):

    registry = {}

    name = None

    DEFAULTS = {}

    REQUIRED = []

    def __init__(self, features=None, config=None, context=None):
        super().__init__(framework="tflite", config=config, context=context)
        self.model = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.name, str)
        cls.registry[cls.name] = cls

    def load_model(self, model):
        self.model = model
