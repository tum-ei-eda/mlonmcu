"""Flow module for frameworks and backend."""

import mlonmcu
from mlonmcu.flow.tflite.backend.tflmc import TFLMCBackend
from mlonmcu.flow.tflite.backend.tflmi import TFLMIBackend

from mlonmcu.flow.tflite.framework import TFLiteFramework
from mlonmcu.flow.tvm.backend.tvmaot import TVMAOTBackend
from mlonmcu.flow.tvm.backend.tvmcg import TVMCGBackend
from mlonmcu.flow.tvm.backend.tvmrt import TVMRTBackend
from mlonmcu.flow.tvm.framework import TVMFramework

SUPPORTED_FRAMEWORKS = {
    "tflite": TFLiteFramework,
    "tvm": TVMFramework,
}

SUPPORTED_TFLITE_BACKENDS = {
    "tflmc": TFLMCBackend,
    "tflmi": TFLMIBackend,
}

SUPPORTED_TVM_BACKENDS = {
    "tvmaot": TVMAOTBackend,
    "tvmrt": TVMRTBackend,
    "tvmcg": TVMCGBackend,
}

SUPPORTED_FRAMEWORK_BACKENDS = {
    "tflite": SUPPORTED_TFLITE_BACKENDS,
    "tvm": SUPPORTED_TVM_BACKENDS,
}

SUPPORTED_BACKENDS = {**SUPPORTED_TFLITE_BACKENDS, **SUPPORTED_TVM_BACKENDS}


def get_available_backend_names():
    """Return all available backend names."""
    return list(SUPPORTED_BACKENDS.keys())
