import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
class InvertedResidual(torch.nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()

		self.stride = stride
		self.use_residual = stride==1 and inp==oup
		hidden_dim = int(round(inp*expand_ratio))
		norm_layer = nn.BatchNorm2d

		self.conv1 = Conv2dNormActivation(in_channels=inp, out_channels=hidden_dim, kernel_size=1,
									stride=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
		self.conv2 = Conv2dNormActivation(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
									stride=stride, padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
		self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0)
		
		self.bn1 = norm_layer(oup)

		self.conv = nn.Sequential(*[
			self.conv1,
			self.conv2,
			self.conv3,
			self.bn1
		])


	def forward(self, x):
		output = self.conv(x)

		if self.use_residual:
			output = x + output

		return output

class MobileNetv2(torch.nn.Module):
	def __init__(self, num_classes):
		super(MobileNetv2, self).__init__()

		inverted_residual_setting = [
			# expansion factor, channel, n depth, stride
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, bias=False)		
		last_output_shape = 32
		inverted_residual_layers = []
		for t, c, n, s in inverted_residual_setting:
			for i in range(n):
				if i==0:
					inverted_residual_layers.append(InvertedResidual(inp=last_output_shape, oup=c, stride=s, expand_ratio=t))
				else:
					inverted_residual_layers.append(InvertedResidual(inp=c, oup=c, stride=1, expand_ratio=t))
				last_output_shape = c

		self.inverted_residual_layers = nn.Sequential(*inverted_residual_layers)
		self.conv2 = nn.Conv2d(in_channels=last_output_shape, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False)
		self.avg_pool = nn.AvgPool2d(kernel_size=7)
		# building classifier
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(1280, num_classes),
		)


	def forward(self, x):
		output = self.conv1(x)
		output = self.inverted_residual_layers(output)
		output = self.conv2(output)
		output = self.avg_pool(output)
		output = torch.flatten(output, 1)
		output = self.classifier(output)
		return output
