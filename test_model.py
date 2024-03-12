import torch

from model import MobileNetv2

print(torch._C._get_default_device())

resnet_model = MobileNetv2(num_classes=10)

print(resnet_model)

# Create a tensor filled with zeros of size 3x224x224 (3 channels for RGB)
dummy_image = torch.zeros(3, 224, 224)
# Reshape the tensor to match the expected input shape for most convolutional models
dummy_image = dummy_image.unsqueeze(0)

output = resnet_model.forward(dummy_image)

print(output)