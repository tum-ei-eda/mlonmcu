import torch
from tflite_like_quantizer import TFLiteLikeQuantizer
from example_quantizer import ExampleQuantizer
import resnet8_fp32

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.extension.export_util.utils import save_pte_program
from executorch.exir import ExecutorchBackendConfig, EdgeCompileConfig, to_edge_transform_and_lower, to_edge

# quantizer = TFLiteLikeQuantizer()
quantizer = ExampleQuantizer()

model = resnet8_fp32.ModelUnderTest.eval()

example_inputs = resnet8_fp32.ModelInputs

exported_program = torch.export.export(model, example_inputs, strict=True)

model = exported_program.module()
model_fp32 = model

prepared = prepare_pt2e(model, quantizer)


def get_dummy_calibration_data(input_shape, num_batches=32, device="cpu"):
    """
    Generates dummy calibration data with realistic activation statistics.

    Args:
        input_shape: e.g. (1, 3, 224, 224)
        num_batches: number of calibration batches
    """

    calibration_data = []

    for _ in range(num_batches):
        # Normal distribution approximates real NN activations well
        data = torch.randn(input_shape, device=device)

        # Optional: clamp to realistic range
        data = torch.clamp(data, -3.0, 3.0)

        calibration_data.append(data)

    return calibration_data


print("example_inputs[0]", example_inputs[0], type(example_inputs[0]), dir(example_inputs[0]))

calibration_data = [example_inputs[0]]

# calibration
for batch in calibration_data:
    prepared(batch)

quantized_model = convert_pt2e(prepared)
exported_program = torch.export.export(model, example_inputs, strict=True)

# quantized_model = result of convert_pt2e(...)

# edge = to_edge(exported_program)
compile_config = EdgeCompileConfig(_check_ir_validity=False)
edge = to_edge_transform_and_lower(exported_program, compile_config=compile_config)

exec_prog = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

save_pte_program(exec_prog, "resnet8_quant.pte")

# Save to file
# with open("resnet8_quant.pte", "wb") as f:
#     f.write(edge_program.to_executorch().buffer)

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    quantized_model,
    dummy,
    "resnet8_quant.onnx",
    opset_version=17,
)
