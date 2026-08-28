"""Built-in torch model definitions for the Torch frontend."""

from .models import MODELS
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.models import MODEL_NAME_TO_MODEL

# from enum import Enum
#
#
# class ModelEnum(str, Enum):
#     Mul = "mul"
#     Linear = "linear"
#     Add = "add"
#     AddMul = "add_mul"
#     Softmax = "softmax"
#     Conv1d = "conv1d"
#     Dl3 = "dl3"
#     Edsr = "edsr"
#     EmformerTranscribe = "emformer_transcribe"
#     EmformerPredict = "emformer_predict"
#     EmformerJoin = "emformer_join"
#     Llama2 = "llama2"
#     Llama = "llama"
#     Llama32VisionEncoder = "llama3_2_vision_encoder"
#     Lstm = "lstm"
#     MobileBert = "mobilebert"
#     Mv2 = "mv2"
#     Mv2Untrained = "mv2_untrained"
#     Mv3 = "mv3"
#     Vit = "vit"
#     W2l = "w2l"
#     Ic3 = "ic3"
#     Ic4 = "ic4"
#     ResNet18 = "resnet18"
#     ResNet50 = "resnet50"
#     Llava = "llava"
#     EfficientSam = "efficient_sam"
#     Qwen25 = "qwen2_5_1_5b"
#     Phi4Mini = "phi_4_mini"
#     SmolLM2 = "smollm2"
#     DeiTTiny = "deit_tiny"
#     Sdpa = "sdpa"
#
#     def __str__(self) -> str:
#         return self.value
#
#
# MODEL_NAME_TO_MODEL = {
#     str(ModelEnum.Mul): ("toy_model", "MulModule"),
#     str(ModelEnum.Linear): ("toy_model", "LinearModule"),
#     str(ModelEnum.Add): ("toy_model", "AddModule"),
#     str(ModelEnum.AddMul): ("toy_model", "AddMulModule"),
#     str(ModelEnum.Softmax): ("toy_model", "SoftmaxModule"),
#     str(ModelEnum.Conv1d): ("toy_model", "Conv1dModule"),
#     str(ModelEnum.Dl3): ("deeplab_v3", "DeepLabV3ResNet50Model"),
#     str(ModelEnum.Edsr): ("edsr", "EdsrModel"),
#     str(ModelEnum.EmformerTranscribe): ("emformer_rnnt", "EmformerRnntTranscriberModel"),
#     str(ModelEnum.EmformerPredict): ("emformer_rnnt", "EmformerRnntPredictorModel"),
#     str(ModelEnum.EmformerJoin): ("emformer_rnnt", "EmformerRnntJoinerModel"),
#     str(ModelEnum.Llama2): ("llama", "Llama2Model"),
#     str(ModelEnum.Llama): ("llama", "Llama2Model"),
#     str(ModelEnum.Llama32VisionEncoder): ("llama3_2_vision", "FlamingoVisionEncoderModel"),
#     # TODO: This take too long to export on both Linux and MacOS (> 6 hours)
#     # "llama3_2_text_decoder": ("llama3_2_vision", "Llama3_2Decoder"),
#     str(ModelEnum.Lstm): ("lstm", "LSTMModel"),
#     str(ModelEnum.MobileBert): ("mobilebert", "MobileBertModelExample"),
#     str(ModelEnum.Mv2): ("mobilenet_v2", "MV2Model"),
#     str(ModelEnum.Mv2Untrained): ("mobilenet_v2", "MV2UntrainedModel"),
#     str(ModelEnum.Mv3): ("mobilenet_v3", "MV3Model"),
#     str(ModelEnum.Vit): ("torchvision_vit", "TorchVisionViTModel"),
#     str(ModelEnum.W2l): ("wav2letter", "Wav2LetterModel"),
#     str(ModelEnum.Ic3): ("inception_v3", "InceptionV3Model"),
#     str(ModelEnum.Ic4): ("inception_v4", "InceptionV4Model"),
#     str(ModelEnum.ResNet18): ("resnet", "ResNet18Model"),
#     str(ModelEnum.ResNet50): ("resnet", "ResNet50Model"),
#     str(ModelEnum.Llava): ("llava", "LlavaModel"),
#     str(ModelEnum.EfficientSam): ("efficient_sam", "EfficientSAM"),
#     str(ModelEnum.Qwen25): ("qwen2_5", "Qwen2_5Model"),
#     str(ModelEnum.Phi4Mini): ("phi_4_mini", "Phi4MiniModel"),
#     str(ModelEnum.SmolLM2): ("smollm2", "SmolLM2Model"),
#     str(ModelEnum.DeiTTiny): ("deit_tiny", "DeiTTinyModel"),
#     str(ModelEnum.Sdpa): ("toy_model", "SdpaModule"),
# }
#
__all__ = ["MODEL_NAME_TO_MODEL", "EagerModelFactory", "MODELS"]
