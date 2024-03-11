import torch
from torch import nn

class Bottleneck(torch.nn.Module):
	def __init__(self, channel_size, stride=1 ,depth=1):
		super(Bottleneck, self).__init__()

		self.conv1 = nn.Conv2d(out_channels=channel_size, kernel_size=1, stride=stride, padding=3)


	def forward(self, x):
		pass

class MobileNetv2(torch.nn.Module):
	def __init__(self, num_classes):
		super(MobileNetv2, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, bias=False)
		



	def forward(self, x):
		pass
