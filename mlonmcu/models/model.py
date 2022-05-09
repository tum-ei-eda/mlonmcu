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
from enum import Enum
from pathlib import Path
from collections import namedtuple

from mlonmcu.config import filter_config

from .metadata import parse_metadata

ModelFormat = namedtuple("ModelFormat", ["value", "extensions"])


class ModelFormats(Enum):
    @property
    def extensions(self):
        return self.value.extensions

    @property
    def extension(self):
        return self.value.extensions[0]

    @classmethod
    def from_extension(cls, ext):
        for fmt in cls:
            if ext in fmt.extensions:
                return fmt
        return None

    NONE = ModelFormat(0, [])
    TFLITE = ModelFormat(1, ["tflite"])
    PACKED = ModelFormat(2, ["tflm"])
    IPYNB = ModelFormat(3, ["ipynb"])
    ONNX = ModelFormat(4, ["onnx"])
    RELAY = ModelFormat(5, ["relay"])


def parse_metadata_from_path(path):
    if Path(path).is_file():
        metadata = parse_metadata(path)
        return metadata
    return None


class Model:

    DEFAULTS = {
        "metadata_path": "definition.yml",
        "support_path": "support",
        "inputs_path": "input",
        "outputs_path": "output",
    }

    def __init__(self, name, paths, config=None, alt=None, formats=ModelFormats.TFLITE):
        self.name = name
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.path]
        self.alt = alt
        self.formats = formats
        if not isinstance(self.formats, list):
            self.formats = [formats]
        self.config = filter_config(config if config is not None else {}, self.name, self.DEFAULTS, [])
        self.metadata = parse_metadata_from_path(self.metadata_path)

    @property
    def metadata_path(self):
        return self.config["metadata_path"]

    @property
    def support_path(self):
        return self.config["support_path"]

    @property
    def inputs_path(self):
        # TODO: fall back to metadata
        return self.config["inputs_path"]

    @property
    def outputs_path(self):
        # TODO: fall back to metadata
        return self.config["outputs_path"]

    def __repr__(self):
        if self.alt:
            return f"Model({self.name},alt={self.alt})"
        return f"Model({self.name})"
