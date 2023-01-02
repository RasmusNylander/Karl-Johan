import medmnist
import torch
from matplotlib import pyplot as plt
from medmnist import INFO, Evaluator

import monai
from monai.transforms import Compose, RandRotate90, ScaleIntensity
from torch import Tensor, randint
from tqdm import tqdm

BATCH_SIZE = 32
NUM_EPOCHS = 5
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

rows, cols = 5, 5
stride = 5
image, label = train_dataset[randint(0, len(train_dataset) - 1, [1])]
plt.figure("image", (7, 7))
print(f"label: {label[0]}")
for row in range(rows):
	for col in range(cols):
		plt.subplot(rows, cols, row * cols + col + 1)
		plt.imshow(image[0][row * cols + col], cmap="gray")
		plt.axis("off")
plt.show()


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
		inputs: Tensor = inputs.to(device, dtype=torch.float32)
		targets: Tensor = targets.to(device, dtype=torch.float32)

		optimizer.zero_grad()
		outputs: Tensor = model(inputs)

		targets = targets.to(device, torch.float32)
		loss = loss_function(outputs, targets)

		loss.backward()
		optimizer.step()