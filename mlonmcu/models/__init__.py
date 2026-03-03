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
    PTEFrontend,
    TorchPickleFrontend,
    TorchPythonFrontend,
    TorchExportedFrontend,
    MLIRFrontend,
    RelayFrontend,
    PaddleFrontend,
    ExampleFrontend,
    EmbenchFrontend,  # 1.0
    EmbenchIoTFrontend,  # 2.0
    EmbenchDSPFrontend,
    TaclebenchFrontend,
    PolybenchFrontend,
    CoremarkFrontend,
    DhrystoneFrontend,
    MathisFrontend,
    MibenchFrontend,
    LayerGenFrontend,
    OpenASIPFrontend,
    RVVBenchFrontend,
    ISSBenchFrontend,
    CryptoBenchFrontend,
    CmsisDSPFrontend,
    CmsisNNFrontend,
)

SUPPORTED_FRONTENDS = {}


def register_frontend(frontend_name, frontend_cls=None, override=False):
    """Register a frontend class under the given name.

    Can be used directly or as a decorator.
    """

    def _register(cls):
        if frontend_name in SUPPORTED_FRONTENDS and not override:
            raise RuntimeError(f"Frontend {frontend_name} is already registered")
        SUPPORTED_FRONTENDS[frontend_name] = cls
        return cls

    if frontend_cls is None:
        return _register
    return _register(frontend_cls)


def get_frontends():
    """Return the registered frontends."""
    return SUPPORTED_FRONTENDS


register_frontend("tflite", TfLiteFrontend)
register_frontend("relay", RelayFrontend)
register_frontend("packed", PackedFrontend)
register_frontend("onnx", ONNXFrontend)
register_frontend("pte", PTEFrontend)
register_frontend("torch_pickle", TorchPickleFrontend)
register_frontend("torch_python", TorchPythonFrontend)
register_frontend("torch_exported", TorchExportedFrontend)
register_frontend("mlir", MLIRFrontend)
register_frontend("pb", PBFrontend)
register_frontend("paddle", PaddleFrontend)
register_frontend("example", ExampleFrontend)
register_frontend("embench", EmbenchFrontend)
register_frontend("embench_iot", EmbenchIoTFrontend)
register_frontend("embench_dsp", EmbenchDSPFrontend)
register_frontend("taclebench", TaclebenchFrontend)
register_frontend("coremark", CoremarkFrontend)
register_frontend("dhrystone", DhrystoneFrontend)
register_frontend("polybench", PolybenchFrontend)
register_frontend("mathis", MathisFrontend)
register_frontend("mibench", MibenchFrontend)
register_frontend("layergen", LayerGenFrontend)
register_frontend("openasip", OpenASIPFrontend)
register_frontend("rvv_bench", RVVBenchFrontend)
register_frontend("iss_bench", ISSBenchFrontend)
register_frontend("crypto_bench", CryptoBenchFrontend)
register_frontend("cmsis_dsp", CmsisDSPFrontend)
register_frontend("cmsis_nn", CmsisNNFrontend)

__all__ = [
    "print_summary",
    "TfLiteFrontend",
    "PackedFrontend",
    "ONNXFrontend",
    "MLIRFrontend",
    "TorchFrontend",
    "PBFrontend",
    "LayerGenFrontend",
    "SUPPORTED_FRONTENDS",
    "register_frontend",
    "get_frontends",
]
