from monai.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch
from torch import Tensor, device as Device


def train_one_epoch(model, dataloader: DataLoader, loss_function: _Loss, optimizer: Optimizer, device: Device, writer, logging_offset: int) -> float:
	model.train()
	num_training_batches = len(dataloader)
	train_loss: Tensor = torch.empty(num_training_batches, device=device)
	for batch_index, (inputs, targets) in enumerate(dataloader):
		optimizer.zero_grad()
		outputs: Tensor = model(inputs.to_device(device))
		targets: Tensor = targets.squeeze(1).long().to(device)

		loss = loss_function(outputs, targets)
		train_loss[batch_index] = loss.item()
		writer.add_scalar('train_loss_logs', loss.item(), logging_offset + batch_index)

		loss.backward()
		optimizer.step()

	return sum(train_loss) / len(train_loss)