from math import sqrt, ceil

import medmnist
import torch
from matplotlib import pyplot as plt
from medmnist import INFO

import monai
from monai.transforms import Compose, RandRotate90, ScaleIntensity
from torch import Tensor, randint
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy(predictions: Tensor, labels: Tensor):
	return (predictions == labels).sum().item() / labels.numel()


def plot_image(image: Tensor):
	num_slices = image.shape[0]
	rows = ceil(sqrt(num_slices))
	cols = ceil(num_slices / rows)

	image_width = cols * (image.shape[2] + 1) / 100
	image_height = rows * (image.shape[1] + 1) / 100

	plt.figure("image", (image_width, image_height))
	for row in range(rows):
		for col in range(cols):
			if row * cols + col >= num_slices:
				plt.show()
				return
			plt.subplot(rows, cols, row * cols + col + 1)
			plt.imshow(image[row * cols + col], cmap="gray")
			plt.axis("off")
	plt.show()


class TestResult:
	def __init__(self, loss: float, accuracy: float):
		self.loss = loss
		self.accuracy = accuracy


def test(model, dataloader: DataLoader) -> TestResult:
	model.eval()
	with torch.no_grad():
		num_batches: int = len(dataloader)
		loss: Tensor = torch.empty(num_batches, device=device)
		accuracy_score: Tensor = torch.empty(num_batches, device=device)

		for batch_index, (inputs, targets) in enumerate(dataloader):
			inputs: Tensor = inputs.to(device, dtype=torch.float32)
			targets: Tensor = targets.to(device, dtype=torch.float32)

			outputs: Tensor = model(inputs)

			loss[batch_index] = loss_function(outputs, targets).item()
			accuracy_score[batch_index] = accuracy(outputs, targets)

		return TestResult(loss.mean().item(), accuracy_score.mean().item())


def train_one_epoch(model, train_loader: DataLoader, validation_loader: DataLoader) -> (float, float, float, float):
	model.train()
	num_training_batches = len(train_loader)
	train_loss: Tensor = torch.empty(num_training_batches, device=device)
	train_accuracy: Tensor = torch.empty(num_training_batches, device=device)
	for batch_index, (inputs, targets) in enumerate(train_loader):
		# forward + backward + optimize
		inputs: Tensor = inputs.to(device, dtype=torch.float32)
		targets: Tensor = targets.to(device, dtype=torch.float32)

		optimizer.zero_grad()
		outputs: Tensor = model(inputs)

		loss = loss_function(outputs, targets)

		train_loss[batch_index] = loss.item()
		train_accuracy[batch_index] = accuracy(outputs, targets)

		loss.backward()
		optimizer.step()

	model.eval()
	with torch.no_grad():
		num_validation_batches: int = len(validation_loader)
		validation_loss: Tensor = torch.empty(num_validation_batches, device=device)
		validation_accuracy: Tensor = torch.empty(num_validation_batches, device=device)
		for batch_index, (inputs, targets) in enumerate(validation_loader):
			inputs: Tensor = inputs.to(device, dtype=torch.float32)
			targets: Tensor = targets.to(device, dtype=torch.float32)

			outputs: Tensor = model(inputs)

			loss = loss_function(outputs, targets)

			validation_loss[batch_index] = loss.item()
			validation_accuracy[batch_index] = accuracy(outputs, targets)

	return train_loss.mean().item(), train_accuracy.mean().item(), validation_loss.mean().item(), validation_accuracy.mean().item()


if __name__ == "__main__":
	BATCH_SIZE = 32
	NUM_EPOCHS = 10
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	dataset_name: str = "organmnist3d"
	download: bool = True

	info = INFO[dataset_name]
	num_classes = len(info["label"])
	DataClass = getattr(medmnist, info['python_class'])

	train_dataset = DataClass(split='train',  download=download, transform=Compose([ScaleIntensity(), RandRotate90()]))
	val_dataset = DataClass(split='val', download=download, transform=Compose([ScaleIntensity()]))
	test_dataset = DataClass(split='test', download=download, transform=Compose([ScaleIntensity()]))
	train_loader = monai.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = monai.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
	test_loader = monai.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	# random_image = train_dataset[randint(0, len(train_dataset) - 1, [1])][0][0]
	# plot_image(random_image)

	model = torch.nn.Sequential(
		torch.nn.Dropout(0.1),
		torch.nn.Conv3d(1, 8, kernel_size=3, stride=2),
		torch.nn.ReLU(),
		torch.nn.Dropout(0.1),
		torch.nn.Conv3d(8, 16, kernel_size=3, stride=2),
		torch.nn.ReLU(),
		torch.nn.Dropout(0.1),
		torch.nn.Conv3d(16, 32, kernel_size=3, stride=2),
		torch.nn.ReLU(),
		torch.nn.Dropout(0.1),
		torch.nn.MaxPool3d(kernel_size=2),
		torch.nn.Flatten(),
		torch.nn.Linear(32, 1),
		torch.nn.Sigmoid()
	).to(device)
	optimizer = torch.optim.Adam(model.parameters(), 1e-3)
	loss_function = torch.nn.BCEWithLogitsLoss()

	for epoch in tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch"):
		train_loss_mean, train_accuracy_mean, validation_loss_mean, validation_accuracy_mean = \
			train_one_epoch(model, train_loader, val_loader)
		# print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train loss: {train_loss_mean:.4f} - Train accuracy: {train_accuracy_mean:.4f} - Validation loss: {validation_loss_mean:.4f} - Validation accuracy: {validation_accuracy_mean:.4f}")

	test_result: TestResult = test(model, test_loader)
	print(f"Test loss: {test_result.loss:.4f} - Test accuracy: {test_result.accuracy:.4f}")