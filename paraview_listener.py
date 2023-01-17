import glob
import os
import socket
from enum import IntEnum

import numpy as np
import torch
from tifffile import tifffile
from torch import Tensor
from torch.nn import functional as F

from create_dataloader import Dataset, Label, MNInSecTVariant, SplitType
from model_picker import ModelType, get_model_name

MODELS_ROOT = "./models"
DATA_PATH = "./datasets/MNInSecT/"
ATTENTION_MAPS_ROOT = "D:/attention_maps"


def scale_image(image: Tensor, new_size) -> Tensor:
    return F.interpolate(image.unsqueeze(dim=0).unsqueeze(dim=0), new_size)[0][0]


def combine_image(image: Tensor, true_map: Tensor, predicted_map: Tensor):
    true_map = scale_image(true_map, image.squeeze().shape)
    true_map[true_map < 0.2] = 0

    predicted_map = scale_image(predicted_map, image.squeeze().shape)
    predicted_map = true_map
    predicted_map[predicted_map < 0.2] = 0

    image = image / image.max()
    image = image[0]
    image[image < 0.1] = 0

    opacity = torch.ones(image.shape, dtype=torch.float32)
    opacity[image != 0] -= 0.2
    opacity[true_map != 0] -= 0.1
    opacity[predicted_map != 0] -= 0.1

    combined = torch.stack([image, predicted_map, true_map, opacity], dim=-1).numpy()
    combined_as_uint8 = (combined * 255).astype(np.uint8)
    return combined_as_uint8


def empty_folder(path: str) -> None:
    contents = glob.glob(f"{path}/*")
    for file in contents:
        os.remove(file)


class Command(IntEnum):
    NextImage = 0,
    PreviousImage = 1,
    NextLayer = 2,
    PreviousLayer = 3,
    NextScale = 4,
    PreviousScale = 5,
    NextModel = 6,
    PreviousModel = 7,
    NextAugmentation = 8,
    PreviousAugmentation = 9,
    RandomImage = 10,

    @staticmethod
    def from_int(i: int) -> "Command":
        return next(command for command in Command if command.numerator == i)


class Configuration:
    def __init__(self, dataset_variant: MNInSecTVariant, model_type: ModelType, layer: int, image_id: int):
        self.dataset_variant = dataset_variant
        self.model_type = model_type
        self.layer = layer
        self.image_id = image_id

        self.dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, variant=self.dataset_variant)

    def parse_input(self, input: bytes) -> bool:
        assert len(Command) == 11
        command = Command.from_int(input[0])
        match command:
            case Command.NextImage:
                self.image_id = (self.image_id + 1) % len(self.dataset)
            case Command.PreviousImage:
                self.image_id = (self.image_id - 1) % len(self.dataset)
            case Command.NextLayer:
                self.layer += 1
                if self.layer > 4:
                    self.layer = 1
            case Command.PreviousLayer:
                self.layer -= 1
                if self.layer <= 0:
                    self.layer = 4
            case Command.NextScale:
                self.dataset_variant = MNInSecTVariant(self.dataset_variant.augmentation, self.dataset_variant.scale.next())
                self.dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, variant=self.dataset_variant)
            case Command.PreviousScale:
                self.dataset_variant = MNInSecTVariant(self.dataset_variant.augmentation, self.dataset_variant.scale.previous())
                self.dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, variant=self.dataset_variant)
            case Command.NextModel:
                self.model_type = self.model_type.next()
            case Command.PreviousModel:
                self.model_type = self.model_type.previous()
            case Command.NextAugmentation:
                self.dataset_variant = MNInSecTVariant(self.dataset_variant.augmentation.next(), self.dataset_variant.scale)
                self.dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, variant=self.dataset_variant)
            case Command.PreviousAugmentation:
                self.dataset_variant = MNInSecTVariant(self.dataset_variant.augmentation.previous(), self.dataset_variant.scale)
                self.dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, variant=self.dataset_variant)
            case Command.RandomImage:
                self.image_id = np.random.randint(0, len(self.dataset))
        return True


variants = [ variant for variant in MNInSecTVariant ]
model_types = [ model_type for model_type in ModelType ]
config = Configuration(dataset_variant=variants[0], model_type=model_types[0], layer=1, image_id=0)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(5)
while True:
    print("Waiting for connection...")
    with server_socket.accept()[0] as connection:

        config_changed = config.parse_input(connection.recv(2))
        if not config_changed:
            connection.send(b"No change")
            continue

        empty_folder("./combined")

        image_name = config.dataset.get_name_of_image(config.image_id)
        true_label = Label.from_abbreviation(image_name[:2].upper())
        insect_image = config.dataset[config.image_id][0]
        model_name = get_model_name(config.model_type, config.dataset_variant)

        true_label_path = os.path.join(ATTENTION_MAPS_ROOT, model_name, f"layer{config.layer}", image_name,
                                       f"{true_label.abbreviation}*.tif")
        true_map_filename = glob.glob(true_label_path)[0]
        true_map = torch.from_numpy(tifffile.imread(true_map_filename))

        prediction_label_path = os.path.join(ATTENTION_MAPS_ROOT, model_name, f"layer{config.layer}", image_name,
                                             f"*prediction*.tif")
        prediction_map_filename = glob.glob(prediction_label_path)[0]
        prediction_map = torch.from_numpy(tifffile.imread(prediction_map_filename))

        tifffile.imwrite(f"./combined/{image_name[:6]}, {model_name} layer {config.layer}.tif",
                         combine_image(insect_image, true_map, prediction_map))
        connection.send(b"Done")
        print("Done")
