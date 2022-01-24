from mlonmcu.models.lookup import print_summary
from .frontend import TfLiteFrontend, PackedFrontend, ONNXFrontend

SUPPORTED_FRONTENDS = {
    "tflite": TfLiteFrontend,
    "packed": PackedFrontend,
    "onnx": ONNXFrontend,
}  # TODO: use registry instead
