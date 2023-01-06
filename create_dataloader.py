import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms



class dataset(Dataset):
    def __init__(self, data_path, train, seed=42):
        if train:
            self.image_paths = list(
                pd.read_csv(data_path + "/train.csv", names=["files"], header=0).files
            )
        else:
            self.image_paths = list(
                pd.read_csv(data_path + "/test.csv", names=["files"], header=0).files
            )

        self.image_paths = [f"{data_path}/{path}" for path in self.image_paths]

        self.image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        self.image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(self.image_classes)}
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = io.imread(image_path)
        image = np.expand_dims(image, 0)

        X = torch.Tensor(image)

        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        temp = torch.zeros(len(self.image_classes))
        temp[y] = 1
        y = temp

        return X, y

    def get_image_paths(self):
        return self.image_paths

    def get_image_classes(self):
        return self.image_classes

    def get_name_to_label(self):
        return self.name_to_label


def make_dataloaders(
    batch_size=16,
    seed=42,
    data_path="../datasets/sorted_downscaled",
    num_workers=4,
    pin_memory=True,
):
    """
    Creates a train and test dataloader with a variable batch size and image shape.
    And using a weighted sampler for the training dataloader to have balanced mini-batches when training.
    """
    train_set = dataset(data_path=data_path, train=True, seed=seed)
    test_set = dataset(data_path=data_path, train=False, seed=seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=np.random.seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=np.random.seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
