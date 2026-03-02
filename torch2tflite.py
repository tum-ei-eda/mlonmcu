import litert_torch
import numpy
import torch
import torchvision

resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)
torch_output = resnet18(*sample_inputs)

edge_model = litert_torch.convert(resnet18.eval(), sample_inputs)

edge_output = edge_model(*sample_inputs)

edge_model.export('resnet.tflite')
