import os
import sys
import time
import argparse

import torch
from monai.data import DataLoader
from torch import Tensor, device as Device
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchmetrics.functional.classification import multiclass_auroc
from tqdm import trange

from create_dataloader import DatasetScale, Augmentation, MNInSecTVariant, make_dataloaders
from accuracy import accuracy
import wandb
from logging_wb import init_logging, log_test_result
from model_picker import ModelType, get_model, get_model_name, get_pretrained


def train_one_epoch(model, dataloader: DataLoader, loss_function: _Loss, optimizer: Optimizer, device: Device) -> float:
    model.train()
    num_training_batches = len(dataloader)
    train_loss: Tensor = torch.empty(num_training_batches, device=device)
    for batch_index, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs: Tensor = model(inputs.to(device))
        targets: Tensor = targets.squeeze(1).to(device)

        loss = loss_function(outputs, targets)
        train_loss[batch_index] = loss.item()

        loss.backward()
        optimizer.step()

    return sum(train_loss) / len(train_loss)


class TestResult:
    def __init__(self, loss: float, accuracy: float, auc: Tensor):
        self.loss = loss
        self.acc = accuracy
        self.auc = auc

    def __str__(self):
        return f"Test loss: {self.loss}, AUC: {self.auc}, ACC: {self.acc}"


def test(model, dataloader: DataLoader, loss_function: _Loss, device: Device) -> TestResult:
    num_classes = len(dataloader.dataset.get_image_classes())
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
            area_under_curve[:, batch_index] = multiclass_auroc(outputs, targets.argmax(dim=1), num_classes=num_classes)

        return TestResult(loss.mean().item(), accuracy_score.mean().item(), area_under_curve.mean(dim=1))

def main(data_path: str, output_root: str, model_pick: ModelType, dataset_variant: MNInSecTVariant, batch_size: int, num_epochs: int, enable_logging: bool, run_log_prefix: str):
    model_name = get_model_name(model_pick, dataset_variant)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, test_loader = make_dataloaders(
        batch_size=batch_size,
        seed=69420,
        data_path=data_path,
        transforms=True,
        pin_memory=False,
        as_rgb=False,
        variant=dataset_variant
    )
        
    num_classes = len(train_loader.dataset.get_image_classes())
    name_to_label = train_loader.dataset.get_name_to_label()
    label_to_name = {v: k for k, v in name_to_label.items()}

    learning_rate = 1e-3
    milestones = [0.1 * num_epochs, 0.8 * num_epochs]
    gamma = 0.1

    model = get_model(model_pick).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    loss_function = torch.nn.CrossEntropyLoss()

    if enable_logging:
        init_logging(f"{run_log_prefix} {model_type.name}", learning_rate, num_epochs, batch_size, model_pick, dataset_variant.scale, model)

    t = time.strftime("%y%m%d_%H%M%S")
    output_path = os.path.join(output_root, f'{run_log_prefix} {model_type.name}_{t}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    best_loss_abs = sys.float_info.max
    best_model_state = model.state_dict().copy()
    log_offset = 0
    with trange(num_epochs, unit="epoch", desc="Epoch 0 – Best AUC: 0 – Best ACC: 0") as progress_bar:
        for epoch in progress_bar:
            train_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)

            train_metrics = test(model, train_loader, loss_function, device)
            validation_metrics = test(model, validation_loader, loss_function, device)

            scheduler.step()
            log_offset += len(train_loader)

            if enable_logging:
                log_test_result(train_metrics, "(train) ")
                log_test_result(validation_metrics, "(validation) ")

            if abs(validation_metrics.loss) < best_loss_abs:
                best_loss_abs = abs(validation_metrics.loss)
                best_model_state = model.state_dict().copy()
                state = {
                    "net": best_model_state,
                }

                path = os.path.join(output_path, f"{model_name}.pth")
                torch.save(state, path)

                progress_bar.set_description(f"Epoch {epoch} – Best AUC: {validation_metrics.auc.mean().item():.5} – Best ACC: {validation_metrics.acc:.5}")

    model = get_pretrained(model_pick, dataset_variant, output_path).to(device)

    train_metrics = test(model, train_loader,  loss_function, device)
    validation_metrics = test(model, validation_loader, loss_function, device)
    test_metrics = test(model, test_loader,  loss_function, device)

    train_log = "train  auc: %.5f  acc: %.5f\n" % (train_metrics.auc.mean(), train_metrics.acc)
    validation_log = f"validation  auc: {validation_metrics.auc.mean():.5f}  acc: {validation_metrics.acc:.5f}\n"
    test_log = "test  auc: %.5f  acc: %.5f\n" % (test_metrics.auc.mean(), test_metrics.acc)

    if enable_logging:
        wandb.log({
            "test accuracy": test_metrics.acc,
            "test area under curve mean":test_metrics.auc.mean()
        })

    log = f"{train_log}\n{validation_log}\n{test_log}\n"
    print(log)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUN model on insect data")
    parser.add_argument("--data_path", default="./datasets/MNInSecT", type=str)
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--model", default="ResNet18", type=str, help='ResNet18, ResNet50, DenseNet, SEResNet50')
    parser.add_argument("--batch_size", default="8", type=int)
    parser.add_argument("--num_epochs", default="100", type=int)
    parser.add_argument("--scale", default="1", type=float)
    parser.add_argument("--wandb_prefix", default=None, type=str, help="prefix for wandb project name. It may be whitespace in which case no prefix is used but it must be specified.")
    parser.add_argument("--no_logging", action="store_true", help="if set, no logging is done")
    parser.add_argument("--dataset_variant", type=str, default="original", help="Variant of MNInSecT to use. Must be 'original', 'masked' or 'threshold'", required=True)

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    model_type = ModelType.parse_from_string(args.model)
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    dataset_variant = MNInSecTVariant(
        DatasetScale.from_float(args.scale),
        Augmentation.parse_from_string(args.dataset_variant)
    )

    enable_logging = not args.no_logging

    match enable_logging, args.wandb_prefix:
        case True, None:
            raise ValueError("No weights and biases prefix specified.")
        case True, _:
            wandb_prefix = args.wandb_prefix.strip()
        case False, _:
            wandb_prefix = ""

    main(data_path, output_path, model_type, dataset_variant, batch_size, num_epochs, enable_logging, wandb_prefix)
