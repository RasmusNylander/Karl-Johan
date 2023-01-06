from monai.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch
from torch import device as Device


def train(model, dataloader: DataLoader, loss_function: _Loss, optimizer: Optimizer, device: Device, writer) -> float:
	total_loss = []
	global iteration

	model.train()
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		optimizer.zero_grad()
		outputs = model(inputs.to(device))

		targets = torch.squeeze(targets, 1).long().to(device)
		loss = loss_function(outputs, targets)

		total_loss.append(loss.item())
		writer.add_scalar('train_loss_logs', loss.item(), iteration)
		iteration += 1

		loss.backward()
		optimizer.step()

	epoch_loss = sum(total_loss) / len(total_loss)
	return epoch_loss