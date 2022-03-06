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
    def __init__(self, t, fix_names=False):
        self.name = t.Name().decode()
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")

        typeLookup = {
            TType.FLOAT32: (4, "float32"),
            TType.UINT8: (1, "uint8"),
            TType.INT8: (1, "int8"),
        }
        self.tysz, self.ty = typeLookup[t.Type()]
        assert self.ty != ""

        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])
        self.shape = shape

        self.size = self.tysz
        for dimSz in self.shape:
            self.size *= dimSz


class ModelInfo:
    def __init__(self, model, fix_names=False):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        self.inTensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            self.inTensors.append(TensorInfo(t, fix_names=fix_names))

        self.outTensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            self.outTensors.append(TensorInfo(t, fix_names=fix_names))


def get_tflite_model_info(model_buf):
    tflModel = tflite.Model.GetRootAsModel(model_buf, 0)

    shapes = {}
    types = {}

    modelInfo = ModelInfo(tflModel)
    for t in modelInfo.inTensors:
        shapes[t.name] = t.shape
        types[t.name] = t.ty
    for t in modelInfo.outTensors:
        shapes[t.name] = t.shape
        types[t.name] = t.ty
    return modelInfo
