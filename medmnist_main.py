import medmnist
import numpy
import torch
from matplotlib import pyplot as plt
from medmnist import INFO

import monai
from monai.transforms import Compose, RandRotate90, ScaleIntensity
from torch import Tensor, randint
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from DummyModel import DummyModel
from accuracy import accuracy

NUMBER_OF_CLASSES = 11


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


def train_one_epoch(model, dataloader: DataLoader) -> TestResult:
    model.train()
    num_training_batches = len(dataloader)
    train_loss: Tensor = torch.empty(num_training_batches, device=device)
    train_accuracy: Tensor = torch.empty(num_training_batches, device=device)
    for batch_index, (inputs, targets) in enumerate(dataloader):
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

    train_result = TestResult(train_loss.mean().item(), train_accuracy.mean().item())

    return train_result


def one_hot_encode(label: numpy.ndarray) -> Tensor:
    return torch.nn.functional.one_hot(
        torch.from_numpy(label).to(dtype=torch.int64), NUMBER_OF_CLASSES
    )[0]


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_EPOCHS = 1002
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name: str = "organmnist3d"
    download: bool = True

    info = INFO[dataset_name]
    num_classes = len(info["label"])
    assert num_classes == NUMBER_OF_CLASSES
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(
        split="train",
        download=download,
        transform=Compose([ScaleIntensity(), RandRotate90(0, 1), RandRotate90(0, 2)]),
        target_transform=one_hot_encode,
    )
    val_dataset = DataClass(
        split="val",
        download=download,
        transform=Compose([ScaleIntensity()]),
        target_transform=one_hot_encode,
    )
    test_dataset = DataClass(
        split="test",
        download=download,
        transform=Compose([ScaleIntensity()]),
        target_transform=one_hot_encode,
    )
    train_loader = monai.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=64,
        num_workers=3,
        persistent_workers=True,
    )
    val_loader = monai.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=64,
        num_workers=3,
        persistent_workers=True,
    )
    test_loader = monai.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=64,
        num_workers=3,
        persistent_workers=True,
    )

    for i in range(3):
        random_image = train_dataset[randint(0, len(train_dataset) - 1, [1])][0][0]
        plot_image(random_image)

    model = DummyModel(NUMBER_OF_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss: Tensor = torch.empty(NUM_EPOCHS, device=device)
    train_accuracy: Tensor = torch.empty(NUM_EPOCHS, device=device)
    validation_loss: Tensor = torch.empty(NUM_EPOCHS, device=device)
    validation_accuracy: Tensor = torch.empty(NUM_EPOCHS, device=device)
    with trange(NUM_EPOCHS, desc="Training", unit="Epoch") as progress_bar:
        for epoch in progress_bar:
            train_result = train_one_epoch(model, train_loader)
            validation_result = test(model, val_loader)

            train_loss[epoch], train_accuracy[epoch] = (
                train_result.loss,
                train_result.accuracy,
            )
            validation_loss[epoch], validation_accuracy[epoch] = (
                validation_result.loss,
                validation_result.accuracy,
            )

            progress_bar.set_description(
                f"Epoch {epoch + 1} ??? Train Loss: {train_result.loss:.4f} Train accuracy: {train_result.accuracy:.4f} - Validation Loss: {validation_result.loss:.4f} Validation accuracy: {validation_result.accuracy:.4f}"
            )

    # plot the loss
    plt.figure()
    plt.plot(train_loss.cpu(), label="train loss")
    plt.plot(validation_loss.cpu(), label="validation loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accuracy.cpu(), label="train accuracy")
    plt.plot(validation_accuracy.cpu(), label="validation accuracy")
    plt.legend()
    plt.show()

    test_result: TestResult = test(model, test_loader)
    print(
        f"Test loss: {test_result.loss:.4f} - Test accuracy: {test_result.accuracy:.4f}"
    )
