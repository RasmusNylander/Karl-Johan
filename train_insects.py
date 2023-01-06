from monai.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch
from torch import Tensor, device as Device
from torchmetrics.functional.classification import multiclass_auroc

from main import accuracy


def train_one_epoch(model, dataloader: DataLoader, loss_function: _Loss, optimizer: Optimizer, device: Device, writer, logging_offset: int) -> float:
	model.train()
	num_training_batches = len(dataloader)
	train_loss: Tensor = torch.empty(num_training_batches, device=device)
	for batch_index, (inputs, targets) in enumerate(dataloader):
		optimizer.zero_grad()
		outputs: Tensor = model(inputs.to_device(device))
		targets: Tensor = targets.squeeze(1).to(device)

		loss = loss_function(outputs, targets)
		train_loss[batch_index] = loss.item()
		writer.add_scalar('train_loss_logs', loss.item(), logging_offset + batch_index)

		loss.backward()
		optimizer.step()

	return sum(train_loss) / len(train_loss)

class TestResult:
	def __init__(self, test_loss: float, accuracy: float, auc: Tensor):
		self.test_loss = test_loss
		self.acc = accuracy
		self.auc = auc

	def __str__(self):
		return f"Test loss: {self.test_loss}, AUC: {self.auc}, ACC: {self.acc}"


def test(model, dataloader: DataLoader, loss_function: _Loss, device: Device) -> TestResult:
	num_classes = model.num_classes
	model.eval()
	with torch.no_grad():
		num_batches: int = len(dataloader)
		loss: Tensor = torch.empty(num_batches, device=device)
		accuracy_score: Tensor = torch.empty(num_batches, device=device)
		area_under_curve: Tensor = torch.empty(num_classes, num_batches, device=device)

		for batch_index, (inputs, targets) in enumerate(dataloader):
			targets: Tensor = targets.to(device)
			outputs: Tensor = model(inputs.to(device))

			loss[batch_index] = loss_function(outputs, targets).item()
			accuracy_score[batch_index] = accuracy(outputs, targets)
			area_under_curve[:, batch_index] = multiclass_auroc(outputs, targets, num_classes=2)

		return TestResult(loss.mean().item(), accuracy_score.mean().item(), area_under_curve.mean(dim=1))