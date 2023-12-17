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
import os
from mlonmcu.flow.backend import Backend
from mlonmcu.models.model import ModelFormats


class TFLMBackend(Backend):
    registry = {}

    name = None

    FEATURES = set()

    DEFAULTS = {}

    REQUIRED = set()

    def __init__(self, features=None, config=None):
        super().__init__(framework="tflm", config=config, features=features)
        self.model = None
        self.supported_formats = [ModelFormats.TFLITE]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert isinstance(cls.name, str)
        cls.registry[cls.name] = cls

    def load_model(self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None):
        self.model = model
        ext = os.path.splitext(model)[1][1:]
        fmt = ModelFormats.from_extension(ext)
        assert fmt == ModelFormats.TFLITE, f"Backend '{self.name}' does not support model format: {fmt.name}"
