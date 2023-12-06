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
import re
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
    PB = ModelFormat(6, ["pb"])
    PADDLE = ModelFormat(7, ["pdmodel"])
    TEXT = ModelFormat(8, ["txt"])


def parse_metadata_from_path(path):
    if Path(path).is_file():
        metadata = parse_metadata(path)
        return metadata
    return None


def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Taken from: https://github.com/apache/tvm/blob/main/python/tvm/driver/tvmc/shape_parser.py

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]" that
        indicates the desired shape for specific model inputs. Colons, forward slashes and dots
        within input_names are supported. Spaces are supported inside of dimension arrays.
    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """

    # Create a regex pattern that extracts each separate input mapping.
    # We want to be able to handle:
    # * Spaces inside arrays
    # * forward slashes inside names (but not at the beginning or end)
    # * colons inside names (but not at the beginning or end)
    # * dots inside names
    pattern = r"(?:\w+\/)?[:\w.]+\:\s*\[\-?\d+(?:\,\s*\-?\d+)*\]"
    input_mappings = re.findall(pattern, inputs_string)
    assert input_mappings
    shape_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and shape.
        name, shape_string = mapping.rsplit(":", 1)
        # Convert shape string into a list of integers or Anys if negative.
        shape = [int(x) for x in shape_string.strip("][").split(",")]
        # Add parsed mapping to shape dictionary.
        shape_dict[name] = shape

    return shape_dict


def parse_type_string(inputs_string):
    pattern = r"(?:\w+\/)?[:\w.]+\:\s*\-?\w+(?:\s*\-?\w+)*"
    input_mappings = re.findall(pattern, inputs_string)
    assert input_mappings
    type_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and type.
        name, type_string = mapping.rsplit(":", 1)
        type_dict[name] = type_string

    return type_dict


class Model:
    DEFAULTS = {
        "metadata_path": "definition.yml",
        "input_shapes": None,
        "output_shapes": None,
        "input_types": None,
        "output_types": None,
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
        self.config = filter_config(config if config is not None else {}, self.name, self.DEFAULTS, [], [])
        self.metadata = parse_metadata_from_path(self.metadata_path)

    @property
    def metadata_path(self):
        return self.config["metadata_path"]

    @property
    def input_shapes(self):
        temp = self.config["input_shapes"]
        if temp:
            if isinstance(temp, str):
                temp = parse_shape_string(temp)
            else:
                assert isinstance(temp, dict)
        return temp

    @property
    def output_shapes(self):
        temp = self.config["output_shapes"]
        if temp:
            if isinstance(temp, str):
                temp = parse_shape_string(temp)
            else:
                assert isinstance(temp, dict)
        return temp

    @property
    def input_types(self):
        temp = self.config["input_types"]
        if temp:
            if isinstance(temp, str):
                temp = parse_type_string(temp)
            else:
                assert isinstance(temp, dict)
        return temp

    @property
    def output_types(self):
        temp = self.config["output_types"]
        if temp:
            if isinstance(temp, str):
                temp = parse_type_string(temp)
            else:
                assert isinstance(temp, dict)
        return temp

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

    @property
    def skip_check(self):
        if len(self.formats) == 0:
            return True
        elif len(self.formats) == 1:
            return self.formats[0] == ModelFormats.TEXT
        else:
            return False

    def __repr__(self):
        if self.alt:
            return f"Model({self.name},alt={self.alt})"
        return f"Model({self.name})"
