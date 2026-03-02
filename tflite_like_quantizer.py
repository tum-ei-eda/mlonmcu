import torch
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    QuantizationConfig,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.quantizer import QuantizationSpec


class TFLiteLikeQuantizer(Quantizer):
    """
    Generic quantizer matching TensorFlow Lite quantization style:
      - int8 activations (per-tensor affine)
      - int8 weights (per-channel symmetric)
      - int32 bias
    """

    def __init__(self):
        super().__init__()

        # Activation quantization spec (int8 asymmetric)
        self.activation_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(
                dtype=torch.int8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            ),
        )

        # Weight quantization spec (int8 symmetric per-channel)
        self.weight_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(
                dtype=torch.int8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            ),
        )

        # Bias spec (int32, scale derived automatically)
        self.bias_spec = QuantizationSpec(
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.int32).min,
            quant_max=torch.iinfo(torch.int32).max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            observer_or_fake_quant_ctr=None,
        )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Annotate supported ops with quantization specs.
        """
        for node in model.graph.nodes:

            if node.op == "call_function":

                if node.target in (
                    torch.ops.aten.linear.default,
                    torch.ops.aten.conv2d.default,
                ):
                    node.meta["quantization_annotation"] = {
                        "input_qspec_map": {
                            node.args[0]: self.activation_spec
                        },
                        "weight_qspec_map": {
                            node.args[1]: self.weight_spec
                        },
                        "bias_qspec_map": (
                            {node.args[2]: self.bias_spec}
                            if len(node.args) > 2 else {}
                        ),
                        "output_qspec": self.activation_spec,
                    }

        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
