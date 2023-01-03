import torch


class Convolution(torch.nn.Module):

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
		super(Convolution, self).__init__()
		self.dropout = torch.nn.Dropout(0.1)
		self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
		self.relu = torch.nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dropout(x)
		x = self.conv(x)
		x = self.relu(x)
		return x


class DummyModel(torch.nn.Module):
	def __init__(self, number_of_classes: int):
		super(DummyModel, self).__init__()
		self.layers = torch.nn.Sequential(
			Convolution(1, 4, 3),
			Convolution(4, 8, 3),
			torch.nn.Dropout(0.1),
			torch.nn.MaxPool3d(kernel_size=3),
			Convolution(8, 16, 3),
			torch.nn.Dropout(0.1),
			torch.nn.MaxPool3d(kernel_size=3),
			torch.nn.Flatten(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, number_of_classes),
			torch.nn.Softmax()
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)
