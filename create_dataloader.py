import itertools
import os
from dataclasses import dataclass
from enum import Enum, IntEnum, auto

import torch
import numpy as np
import pandas as pd
from tifffile import tifffile
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation


class Label(IntEnum):
    Blowfly = 1,
    CurlyWingedFly = 3,
    Pupae = 7,
    Maggot = 5,
    BuffaloBeetleLarvae = 2,
    Mealworm = 6,
    SoliderFlyLarvae = 8,
    Woodlice = 9,
    BlackCricket = 0,
    Grasshopper = 4,

    @staticmethod
    def abbreviation_dict():
        assert len(Label) == 10
        return {
            Label.Blowfly: "BF",
            Label.CurlyWingedFly: "CF",
            Label.Pupae: "PP",
            Label.Maggot: "MA",
            Label.BuffaloBeetleLarvae: "BL",
            Label.Mealworm: "ML",
            Label.SoliderFlyLarvae: "SL",
            Label.Woodlice: "WO",
            Label.BlackCricket: "BC",
            Label.Grasshopper: "GH",
        }

    @property
    def abbreviation(self):
        return self.abbreviation_dict()[self]

    @staticmethod
    def from_abbreviation(abbreviation: str):
        return next(label for label, label_abbreviation in Label.abbreviation_dict().items() if label_abbreviation == abbreviation.upper())


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
        match scale:
            case 0.25:
                return DatasetScale.Scale25
            case 0.5:
                return DatasetScale.Scale50
            case 1.0:
                return DatasetScale.Scale100
            case _:
                raise ValueError(f"Invalid scale: {scale}")

    def __str__(self):
        assert len(DatasetScale) == 3
        if self == DatasetScale.Scale25:
            return "25"
        elif self == DatasetScale.Scale50:
            return "50"
        elif self == DatasetScale.Scale100:
            return "100"

    def next(self):
        assert len(DatasetScale) == 3
        if self == DatasetScale.Scale25:
            return DatasetScale.Scale50
        elif self == DatasetScale.Scale50:
            return DatasetScale.Scale100
        elif self == DatasetScale.Scale100:
            return DatasetScale.Scale25

    def previous(self):
        assert len(DatasetScale) == 3
        if self == DatasetScale.Scale25:
            return DatasetScale.Scale100
        elif self == DatasetScale.Scale50:
            return DatasetScale.Scale25
        elif self == DatasetScale.Scale100:
            return DatasetScale.Scale50


class Augmentation(Enum):
    Original = auto()
    Masked = auto()
    Threshold = auto()

    @staticmethod
    def parse_from_string(string: str):
        assert len(Augmentation) == 3
        match string.lower():
            case "none" | "original" | "orig" | "o":
                return Augmentation.Original
            case "masked" | "mask" | "m":
                return Augmentation.Masked
            case "threshold" | "thresh" | "t":
                return Augmentation.Threshold
            case _:
                raise ValueError(f"Unknown augmentation: {string}")

    def next(self):
        assert len(Augmentation) == 3
        if self == Augmentation.Original:
            return Augmentation.Masked
        elif self == Augmentation.Masked:
            return Augmentation.Threshold
        elif self == Augmentation.Threshold:
            return Augmentation.Original

    def previous(self):
        assert len(Augmentation) == 3
        if self == Augmentation.Original:
            return Augmentation.Threshold
        elif self == Augmentation.Masked:
            return Augmentation.Original
        elif self == Augmentation.Threshold:
            return Augmentation.Masked


class MNInSecTVariantMeta(type):
    def __iter__(cls):
        iterator = itertools.product(Augmentation, DatasetScale)
        for augmentation, scale in iterator:
            yield MNInSecTVariant(augmentation, scale)

@dataclass
class MNInSecTVariant(metaclass=MNInSecTVariantMeta):
    augmentation: Augmentation
    scale: DatasetScale

    @property
    def base_name(self) -> str:
        assert len(DatasetScale) == 3, "Unhandled scale"
        if self.scale == DatasetScale.Scale25:
            return "64x32x32"
        elif self.scale == DatasetScale.Scale50:
            return "128x64x64"
        elif self.scale == DatasetScale.Scale100:
            return "256x128x128"

    @property
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
        return f"{self.base_name}{self.augmentation_suffix}"

    def __str__(self):
        return self.name


class Dataset(TorchDataset):
    def __init__(self, MNInSecT_root: str, variant: MNInSecTVariant, type: SplitType, seed=42, as_rgb=False, transforms=False):

        dataset_path = os.path.join(MNInSecT_root, variant.name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        self.image_paths, self.image_labels = self.dataset_images(MNInSecT_root, type)
        self.image_paths = [os.path.join(dataset_path, path) for path in self.image_paths]
        self.rng = np.random.default_rng(seed=seed)
        self.as_rgb = as_rgb
        self.transforms = transforms
        self.rot = RandomRotation(180)

    @staticmethod
    def dataset_images(MNInSecT_root: str, type: SplitType) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = "train" if type == SplitType.Train else "test" if type == SplitType.Test else "validation"
        files = pd.read_csv(f"{MNInSecT_root}/{file_name}.csv", names=["files"], header=0).files
        labels = [Label.from_abbreviation(abbreviation) for abbreviation in files.map(lambda x: x[:2]).to_list()]
        return files.to_list(), labels

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]

        image = tifffile.imread(image_path)

        image = np.expand_dims(image, 0)
        
        X = torch.Tensor(image)
        if self.transforms:
            X = self.rot(X)
        if self.as_rgb:
            size = X.shape
            X = X.expand([3, size[1], size[2], size[3]])

        y = label.value
        one_hot_encoded = torch.zeros(self.num_classes())
        one_hot_encoded[y] = 1

        return X, one_hot_encoded

    @staticmethod
    def num_classes() -> int:
        return len(Label)

    @staticmethod
    def label_to_name(label: int) -> str:
        return Label(label).abbreviation

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
