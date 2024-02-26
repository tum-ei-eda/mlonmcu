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
from math import sqrt
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
    assert input_mappings, f"Invalid shapes string: {inputs_string} (Expected syntax: 'foo:[1,32,32,3] bar:[10,10]')"
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
    """Parse an input type dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:ty input_name2:ty" that
        indicates the desired type for specific model inputs/outputs. Colons, forward slashes and dots
        within input_names are supported. Spaces are supported inside of dimension arrays.
    Returns
    -------
    type_dict: dict
        A dictionary mapping input names to their type.
    """
    pattern = r"(?:\w+\/)?[:\w.]+\:\s*\-?\w+(?:\s*\-?\w+)*"
    input_mappings = re.findall(pattern, inputs_string)
    assert input_mappings, f"Invalid types string: {inputs_string} (Expected syntax: 'foo:int8 bar:int32')"
    type_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and type.
        name, type_string = mapping.rsplit(":", 1)
        type_dict[name] = type_string

    return type_dict


class Workload:
    DEFAULTS = {}

    def __init__(self, name, config=None, alt=None):
        self.name = name
        self.alt = alt
        self.config = filter_config(config if config is not None else {}, self.name, self.DEFAULTS, set(), set())

    def get_platform_config(self, platform):
        return {}

    def add_platform_config(self, platform, config):
        config.update(self.get_platform_config(platform))

    def get_platform_defs(self, platform):
        return {}

    def add_platform_defs(self, platform, defs):
        defs.update(self.get_platform_defs(platform))


class Model(Workload):
    DEFAULTS = {
        **Workload.DEFAULTS,
        "metadata_path": "definition.yml",
        "input_shapes": None,
        "output_shapes": None,
        "input_types": None,
        "output_types": None,
        "support_path": None,
        "inputs_path": None,
        "outputs_path": None,
        "output_labels_path": None,
    }

    def __init__(self, name, paths, config=None, alt=None, formats=ModelFormats.TFLITE):
        super().__init__(name, config=config, alt=alt)
        self.paths = paths
        if not isinstance(self.paths, list):
            self.paths = [self.path]
        self.formats = formats
        if not isinstance(self.formats, list):
            self.formats = [formats]
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
        value = self.config["support_path"]
        if value is not None:
            if not isinstance(value, Path):
                assert isinstance(value, str)
                value = Path(value)
        return value

    @property
    def inputs_path(self):
        # TODO: fall back to metadata
        value = self.config["inputs_path"]
        if value is not None:
            if not isinstance(value, Path):
                assert isinstance(value, str)
                value = Path(value)
        return value

    @property
    def outputs_path(self):
        # TODO: fall back to metadata
        value = self.config["outputs_path"]
        if value is not None:
            if not isinstance(value, Path):
                assert isinstance(value, str)
                value = Path(value)
        return value

    @property
    def output_labels_path(self):
        # TODO: fall back to metadata
        value = self.config["output_labels_path"]
        if value is not None:
            if not isinstance(value, Path):
                assert isinstance(value, str)
                value = Path(value)
        return value

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


class Program(Workload):
    def __repr__(self):
        if self.alt:
            return f"Program({self.name},alt={self.alt})"
        return f"Program({self.name})"


class ExampleProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["EXAMPLE_BENCHMARK"] = self.name
        return ret


class EmbenchProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["EMBENCH_BENCHMARK"] = self.name
        return ret


class TaclebenchProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["TACLEBENCH_BENCHMARK"] = self.name
        return ret


class PolybenchProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["POLYBENCH_BENCHMARK"] = self.name
        return ret


class MibenchProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["MIBENCH_BENCHMARK"] = self.name
        return ret


class MathisProgram(Program):
    DEFAULTS = {
        "size": 1024,
        # "size": 65536,
    }

    @property
    def size(self):
        value = self.config["size"]
        if isinstance(value, str):
            value = str(value)
        assert isinstance(value, int)
        assert value > 0
        return value

    def get_nargs(self, name):
        return {
            "to_upper": 2,
            "add8": 4,
            "add16": 4,
            "gather_add8": 4,
            "gather_add16": 4,
            "scatter_add8": 4,
            "scatter_add16": 4,
            "dot8": 3,
            "dot16": 3,
            "saxpy8": 5,
            "saxpy16": 5,
            "matmul8": 4,
            "matmul16": 4,
            "matmul8_a": 4,
            "matmul16_a": 4,
            "transposed_matmul8": 4,
            "transposed_matmul16": 4,
            "transposed_matmul8_a": 4,
            "transposed_matmul16_a": 4,
            "transposed_matmul8_b": 4,
            "transposed_matmul16_b": 4,
        }[name]

    def get_elem_size(self, name):
        return {
            "to_upper": 8,
            "add8": 8,
            "add16": 16,
            "gather_add8": 8,
            "gather_add16": 16,
            "scatter_add8": 8,
            "scatter_add16": 16,
            "dot8": 8,
            "dot16": 16,
            "saxpy8": 8,
            "saxpy16": 16,
            "matmul8": 8,
            "matmul16": 16,
            "matmul8": 8,
            "matmul16": 16,
            "matmul8_a": 8,
            "matmul16_a": 16,
            "transposed_matmul8": 8,
            "transposed_matmul16": 16,
            "transposed_matmul8_a": 8,
            "transposed_matmul16_a": 16,
            "transposed_matmul8_b": 8,
            "transposed_matmul16_b": 16,
        }[name]

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["MATHIS_TEST"] = self.name
            ret["MATHIS_NARGS"] = self.get_nargs(self.name)
            ret["MATHIS_ELEM_SIZE"] = self.get_elem_size(self.name)
            ret["MATHIS_SIZE"] = self.size
            ret["MATHIS_N"] = int(sqrt(self.size)) if "matmul" in self.name else self.size
        return ret


class CoremarkProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["COREMARK_ITERATIONS"] = 10
        return ret


class DhrystoneProgram(Program):
    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["DHRYSTONE_ITERATIONS"] = 10000
        return ret


class OpenASIPProgram(Program):
    DEFAULTS = {
        "crc_mode": "both",
    }

    @property
    def crc_mode(self):
        return str(self.config["crc_mode"])

    def get_platform_defs(self, platform):
        ret = {}
        if platform == "mlif":
            ret["OPENASIP_BENCHMARK"] = self.name
            if self.name == "crc":
                ret["OPENASIP_CRC_MODE"] = self.crc_mode
        return ret
