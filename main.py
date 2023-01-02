import medmnist
import torch
from medmnist import INFO, Evaluator

import monai
from torch import Tensor
from torch import utils
from tqdm import tqdm

BATCH_SIZE = 32
NUM_EPOCHS = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_name: str = "organmnist3d"
download: bool = True

info = INFO[dataset_name]
num_classes = len(info["label"])
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train',  download=download)
train_loader = monai.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = torch.nn.Sequential(
	torch.nn.Conv3d(1, 16, kernel_size=16, stride=3),
	torch.nn.ReLU(),
	torch.nn.Flatten(),
	torch.nn.Linear(2000, 1),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
loss_function = torch.nn.BCEWithLogitsLoss()

model.train()

for epoch in range(NUM_EPOCHS):
	model.train()
	for inputs, targets in tqdm(train_loader):
		# forward + backward + optimize
		optimizer.zero_grad()
		tep: Tensor = inputs.to(device, dtype=torch.float32)
		outputs: Tensor = model(tep)

		targets = targets.to(device, torch.float32)
		loss = loss_function(outputs, targets)

		loss.backward()
		optimizer.step()