import glob
from tqdm.contrib import itertools
import os
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import Tensor

from create_dataloader import Dataset, DatasetScale, Label, MNInSecTVariant, SplitType, Augmentation
from model_picker import ModelType, get_model_name

MODELS_ROOT = "./models"
DATA_PATH = "./datasets/MNInSecT/"
ATTENTION_MAPS_ROOT = "D:/attention_maps"
BATCH_SIZE = 1
assert BATCH_SIZE == 1

model_type: ModelType = ModelType.SEResNet50
scale: DatasetScale = DatasetScale.Scale50
dataset_augmentation: Augmentation = Augmentation.Original
# layer = 3
dataset_variant = MNInSecTVariant(dataset_augmentation, scale)


@dataclass
class AttentionMap:
    model: ModelType
    dataset: MNInSecTVariant
    layer: int
    image_name: str
    label: Label

    def fetch(self, root: str) -> Tensor:
        attention_maps_path = os.path.join(root, get_model_name(self.model, self.dataset), f"layer{self.layer}", self.image_name, f"{self.label.abbreviation}*.tif")
        attention_map_filename = glob.glob(attention_maps_path)[0]
        attention_map = torch.from_numpy(tifffile.imread(attention_map_filename))
        return attention_map


def combine_image(predicted_map: Tensor, image: Tensor):
    predicted_map = F.interpolate(predicted_map.unsqueeze(dim=0).unsqueeze(dim=0), image.squeeze().shape)[0]
    predicted_map = predicted_map[0]
    predicted_map[predicted_map < 0.2] = 0

    image = image / image.max()
    image = image[0]
    image[image < 0.1] = 0

    # background = (attention_map == 0) & (image == 0)
    opacity = torch.ones(image.shape)
    # opacity[background] = 1
    opacity[image != 0] -= 0.2
    opacity[predicted_map != 0] -= 0.1

    combined = torch.stack([image, predicted_map, torch.zeros(image.shape), opacity], dim=-1).numpy()
    combined_as_uint8 = (combined * 255).astype(np.uint8)
    return combined_as_uint8



dataset: Dataset = Dataset(MNInSecT_root=DATA_PATH, type=SplitType.Test, seed=69420, as_rgb=False, variant=dataset_variant)

for layer, image_id in itertools.product(range(1, 5), range(len(dataset))):
    image_name = dataset.get_name_of_image(image_id)
    for label in Label:
        attention_map = AttentionMap(model_type, dataset_variant, layer, image_name, label)
        attention_map_data = attention_map.fetch(ATTENTION_MAPS_ROOT)
        insect_image = dataset[image_id][0]

        path = os.path.join("combined", image_name, f"{layer}")
        os.makedirs(path, exist_ok=True)
        tifffile.imwrite(f"{path}/{label.abbreviation}.tif", combine_image(attention_map_data, insect_image))
