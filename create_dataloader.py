import os
from enum import Enum, auto

import glob
import torch
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation
from scipy.ndimage import zoom


class DatasetType(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()


class Dataset(TorchDataset):
    def __init__(self, data_path, type: DatasetType, seed=42, as_rgb=False, transforms=False, scale: float = 1.0):
        assert scale == 1.0 or scale == 0.5 or scale == 0.25, "Scale must be 1.0, 0.5 or 0.25"
        if scale == 1.0:
            scale_path_offset = ""
        if scale == 0.5:
            scale_path_offset = "_128x64x64"
        elif scale == 0.25:
            scale_path_offset = "_64x32x32"

        assert len(DatasetType) == 3
        match type:
            case DatasetType.Train:
                self.image_paths = pd.read_csv(data_path + "/train.csv", names=["files"], header=0).files.tolist()
            case DatasetType.Test:
                self.image_paths = pd.read_csv(data_path + "/test.csv", names=["files"], header=0).files.tolist()
            case DatasetType.Validation:
                self.image_paths = pd.read_csv(data_path + "/validation.csv", names=["files"], header=0).files.tolist()




        self.image_paths = [f"{data_path}{scale_path_offset}/{path}" for path in self.image_paths]
        self.image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        self.image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(self.image_classes)}
        self.rng = np.random.default_rng(seed=seed)
        self.as_rgb = as_rgb
        self.transforms = transforms
        self.rot = RandomRotation(180)

    def __len__(self):
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx):
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


def make_dataloaders(
    batch_size=16,
    seed=42,
    data_path="./datasets/sorted_downscaled",
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    as_rgb=False,
    transforms=False,
    scale: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates a train and test dataloader with a variable batch size and image shape.
    And using a weighted sampler for the training dataloader to have balanced mini-batches when training.
    """
    train_set = Dataset(data_path=data_path, type=DatasetType.Train, seed=seed, as_rgb=as_rgb, transforms=transforms, scale=scale)
    validation_set = Dataset(data_path=data_path, type=DatasetType.Validation, seed=seed, as_rgb=as_rgb, scale=scale)
    test_set = Dataset(data_path=data_path, type=DatasetType.Test, seed=seed, as_rgb=as_rgb, scale=scale)

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
