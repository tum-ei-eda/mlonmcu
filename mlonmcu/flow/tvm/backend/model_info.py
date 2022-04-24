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
import tflite
from tflite.TensorType import TensorType as TType


class TensorInfo:
    def __init__(self, name, shape, dtype, fix_names=False):
        self.name = name
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")
        assert isinstance(shape, (tuple, list))
        self.shape = shape
        assert dtype in ["float32", "uint8", "int8"]
        size_lookup = {
            "float32": 4,
            "uint8": 1,
            "int8": 1,
        }
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
        }
        dtype = type_lookup[t.Type()]
        super().__init__(name, shape, dtype)

class RelayTensorInfo(TensorInfo):
    def __init__(self, t, fix_names=False):
        pass


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


class RelayModelInfo(ModelInfo):
    def __init__(self, text, fix_names=False):
        pass

def get_tflite_model_info(model_buf):
    tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)
    model_info = TfLiteModelInfo(tflite_model)
    return model_info

def get_relay_model_info(mod_text):
    model_info = RelayModelInfo(mod_text)
    return model_info
