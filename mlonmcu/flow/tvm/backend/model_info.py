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
import tflite
from tflite.TensorType import TensorType as TType


class TensorInfo:
    def __init__(self, name, shape, dtype, fix_names=False):
        self.name = name
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")
        assert isinstance(shape, (tuple, list))
        self.shape = shape
        size_lookup = {
            "float32": 4,
            "uint8": 1,
            "int8": 1,
            "int32": 4,
        }
        assert dtype in size_lookup
        self.dtype = dtype
        self.type_size = size_lookup[self.dtype]

    @property
    def size(self):
        ret = self.type_size
        for dim in self.shape:
            ret *= dim
        return ret


class TfLiteTensorInfo(TensorInfo):
    def __init__(self, t, fix_names=False):
        name = t.Name().decode()
        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])

        type_lookup = {
            TType.FLOAT32: "float32",
            TType.UINT8: "uint8",
            TType.INT8: "int8",
            TType.INT32: "int32",
            TType.BOOL: "int8",
        }
        dtype = type_lookup[t.Type()]
        super().__init__(name, shape, dtype, fix_names=fix_names)


class RelayTensorInfo(TensorInfo):
    def __init__(self, name, shape, dtype, fix_names=False):
        super().__init__(name, shape, dtype, fix_names=fix_names)


class ModelInfo:
    def __init__(self, in_tensors, out_tensors, fix_names=False):
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors


class TfLiteModelInfo(ModelInfo):
    def __init__(self, model, fix_names=False):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        in_tensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            in_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))

        out_tensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            out_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))
        super().__init__(in_tensors, out_tensors)


def shape_from_str(shape_str):
    return tuple(map(int, shape_str.replace(" ", "").split(",")))


def parse_relay_main(line):
    input_tensors = []
    output_tensors = []

    input_tensors_strs = re.compile(r"%[a-zA-Z0-9_]+: Tensor\[\((?:\d+)(?:,\s*\d+)*\), (?:[a-zA-Z0-9_]+)\]").findall(
        line
    )
    for input_tensors_str in input_tensors_strs:
        res = re.compile(r"%([a-zA-Z0-9]+): Tensor\[\((\d+(?:, \d+)+)\), ([a-zA-Z0-9_]+)\]").match(input_tensors_str)
        assert res is not None
        groups = res.groups()
        assert len(groups) == 3
        input_name, input_shape_str, input_type = groups
        input_shape = shape_from_str(input_shape_str)
        input_tensor = TensorInfo(input_name, input_shape, input_type)
        input_tensors.append(input_tensor)

    output_tensor_names_str = re.compile(r"output_tensor_names=\[(\".*\")\]").findall(line)
    output_tensor_names = re.compile(r"\"([a-zA-Z0-9_]+)\"").findall(output_tensor_names_str[0])

    output_tensors_str = re.compile(r"-> (.+) {").findall(line)
    output_tensor_strs = re.compile(r"Tensor\[\(\d+(?:, \d+)+\), [a-zA-Z0-9_]+\]").findall(output_tensors_str[0])

    assert len(output_tensor_names) == len(output_tensor_strs)

    for i, output_name in enumerate(output_tensor_names):
        res = re.compile(r"Tensor\[\((\d+(?:, \d+)+)\), ([a-zA-Z0-9_]+)\]").match(output_tensor_strs[i])
        assert res is not None
        groups = res.groups()
        assert len(groups) == 2
        output_shape_str, output_type = groups
        output_shape = shape_from_str(output_shape_str)
        output_tensor = TensorInfo(output_name, output_shape, output_type)
        output_tensors.append(output_tensor)
    return input_tensors, output_tensors


class RelayModelInfo(ModelInfo):
    def __init__(self, mod_text, fix_names=False):
        in_tensors = None
        out_tensors = None
        for line in mod_text.split("\n"):
            if "def @main(" in line:
                in_tensors, out_tensors = parse_relay_main(line)
                break
        assert in_tensors is not None and out_tensors is not None
        super().__init__(in_tensors, out_tensors)


def get_tflite_model_info(model_buf):
    tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)
    model_info = TfLiteModelInfo(tflite_model)
    return model_info


def get_relay_model_info(mod_text):
    model_info = RelayModelInfo(mod_text)
    return model_info
