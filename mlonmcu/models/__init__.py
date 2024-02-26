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
from mlonmcu.models.lookup import print_summary
from .frontend import (
    PBFrontend,
    TfLiteFrontend,
    PackedFrontend,
    ONNXFrontend,
    RelayFrontend,
    PaddleFrontend,
    ExampleFrontend,
    EmbenchFrontend,
    TaclebenchFrontend,
    PolybenchFrontend,
    CoremarkFrontend,
    DhrystoneFrontend,
    MathisFrontend,
    MibenchFrontend,
    LayerGenFrontend,
    OpenASIPFrontend,
)

SUPPORTED_FRONTENDS = {
    "tflite": TfLiteFrontend,
    "relay": RelayFrontend,
    "packed": PackedFrontend,
    "onnx": ONNXFrontend,
    "pb": PBFrontend,
    "paddle": PaddleFrontend,
    "example": ExampleFrontend,
    "embench": EmbenchFrontend,
    "taclebench": TaclebenchFrontend,
    "coremark": CoremarkFrontend,
    "dhrystone": DhrystoneFrontend,
    "polybench": PolybenchFrontend,
    "mathis": MathisFrontend,
    "mibench": MibenchFrontend,
    "layergen": LayerGenFrontend,
    "openasip": OpenASIPFrontend,
}  # TODO: use registry instead

__all__ = [
    "print_summary",
    "TfLiteFrontend",
    "PackedFrontend",
    "ONNXFrontend",
    "PBFrontend",
    "LayerGenFrontend",
    "SUPPORTED_FRONTENDS",
]
