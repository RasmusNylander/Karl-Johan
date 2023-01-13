import os
from dataclasses import dataclass
from enum import Enum, auto

import glob
import torch
import numpy as np
import pandas as pd
from skimage import io
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation


class SplitType(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()


class DatasetScale(Enum):
    Scale25 = auto()
    Scale50 = auto()
    Scale100 = auto()

    def to_float(self) -> float:
        assert len(DatasetScale) == 3
        match self:
            case DatasetScale.Scale25:
                return 0.25
            case DatasetScale.Scale50:
                return 0.5
            case DatasetScale.Scale100:
                return 1.0

    @staticmethod
    def from_float(scale: float):
        assert len(DatasetScale) == 3
        if scale == 0.25:
            return DatasetScale.Scale25
        elif scale == 0.5:
            return DatasetScale.Scale50
        elif scale == 1.0:
            return DatasetScale.Scale100

    def __str__(self):
        assert len(DatasetScale) == 3
        if self == DatasetScale.Scale25:
            return "25"
        elif self == DatasetScale.Scale50:
            return "50"
        elif self == DatasetScale.Scale100:
            return "100"


class Augmentation(Enum):
    Original = auto()
    Masked = auto()
    Threshold = auto()

@dataclass
class MNInSecTVariant:
    augmentation: Augmentation
    scale: DatasetScale

    def base_name(self) -> str:
        assert len(DatasetScale) == 3, "Unhandled scale"
        if self.scale == DatasetScale.Scale25:
            return "64x32x32"
        elif self.scale == DatasetScale.Scale50:
            return "128x64x64"
        elif self.scale == DatasetScale.Scale100:
            return "256x128x128"

    def augmentation_suffix(self) -> str:
        assert len(Augmentation) == 3, "Unhandled augmentation"
        if self.augmentation == Augmentation.Original:
            return ""
        elif self.augmentation == Augmentation.Masked:
            return "_masked"
        elif self.augmentation == Augmentation.Threshold:
            return "_threshold"

    @property
    def name(self) -> str:
        return f"{self.base_name()}{self.augmentation_suffix()}"

    def __str__(self):
        return self.name


class Dataset(TorchDataset):
    def __init__(self, MNInSecT_root: str, variant: MNInSecTVariant, type: SplitType, seed=42, as_rgb=False, transforms=False):

        dataset_path = os.path.join(MNInSecT_root, variant.name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        self.image_paths = self.dataset_images(MNInSecT_root, type)

        self.image_paths = [os.path.join(dataset_path, path) for path in self.image_paths]
        self.image_classes = [
            os.path.split(d)[1] for d in glob.glob(dataset_path + "/*") if os.path.isdir(d)
        ]
        self.image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(self.image_classes)}
        self.rng = np.random.default_rng(seed=seed)
        self.as_rgb = as_rgb
        self.transforms = transforms
        self.rot = RandomRotation(180)

    @staticmethod
    def dataset_images(MNInSecT_root: str, type: SplitType) -> list[str]:
        assert len(SplitType) == 3
        file_name = "train" if type == SplitType.Train else "test" if type == SplitType.Test else "validation"
        return pd.read_csv(f"{MNInSecT_root}/{file_name}.csv", names=["files"], header=0).files.tolist()

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]

        image = io.imread(image_path)

        image = np.expand_dims(image, 0)
        
        X = torch.Tensor(image)
        if self.transforms:
            X = self.rot(X)
        if self.as_rgb:
            size = X.shape
            X = X.expand([3, size[1], size[2], size[3]])
        

        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        temp = torch.zeros(len(self.image_classes))
        temp[y] = 1
        y = temp

        return X, y

    def get_image_paths(self) -> list[str]:
        return self.image_paths

    def get_image_classes(self) -> list[str]:
        return self.image_classes

    def get_name_to_label(self) -> dict:
        return self.name_to_label

    def num_classes(self) -> int:
        return len(self.image_classes)

    def label_to_name(self, label: int) -> str:
        return self.image_classes[label]

    def get_name_of_image(self, idx: int) -> str:
        return self.image_paths[idx].split("/")[-1].split(".")[0]


def make_dataloaders(
    variant: MNInSecTVariant,
    batch_size=16,
    seed=42,
    data_path="./datasets/MNInSecT",
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    as_rgb=False,
    transforms=False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates a train and test dataloader with a variable batch size and image shape.
    And using a weighted sampler for the training dataloader to have balanced mini-batches when training.
    """
    train_set = Dataset(MNInSecT_root=data_path, type=SplitType.Train, seed=seed, as_rgb=as_rgb, transforms=transforms, variant=variant)
    validation_set = Dataset(MNInSecT_root=data_path, type=SplitType.Validation, seed=seed, as_rgb=as_rgb, variant=variant)
    test_set = Dataset(MNInSecT_root=data_path, type=SplitType.Test, seed=seed, as_rgb=as_rgb, variant=variant)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=np.random.seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=np.random.seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=np.random.seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return train_loader, validation_loader, test_loader
