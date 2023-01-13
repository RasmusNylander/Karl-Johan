import glob

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from monai.data.image_reader import nib

from create_dataloader import Dataset, DatasetScale, DatasetType, MNInSecTVariant
from model_picker import ModelType, get_model_name

MODELS_ROOT = "./models"
DATA_PATH = "./datasets/MNInSecT/"
ATTENTION_MAPS_ROOT = "D:/attention_maps"
BATCH_SIZE = 1
assert BATCH_SIZE == 1

model_type: ModelType = ModelType.ResNet18
scale: DatasetScale = DatasetScale.Scale100
dataset_variant: MNInSecTVariant = MNInSecTVariant.Original
layer = 2

model_string_id = get_model_name(model_type, dataset_variant, scale)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
dataset: Dataset = Dataset(MNInSecT_root=DATA_PATH, type=DatasetType.Test, seed=69420, as_rgb=False, scale=scale, variant=dataset_variant)

image, batch_labels = dataset[0]
image_name = dataset.get_name_of_image(0)

attentionmap_paths = glob.glob(f"{ATTENTION_MAPS_ROOT}/{model_string_id}/layer{layer}/{image_name}/*")
prediction_path = [path for path in attentionmap_paths if "prediction" in path][0]
prediction_map = np.array(nib.load(prediction_path).dataobj).transpose(2, 0, 1)
prediction_map = torch.from_numpy(prediction_map)

attention_map = F.interpolate(prediction_map.unsqueeze(dim=0).unsqueeze(dim=0), image.squeeze().shape)[0]
attention_map = attention_map[0]
attention_map[attention_map < 0.2] = 0

image = image / image.max()
image = image[0]
image[image < 0.1] = 0

background = (attention_map == 0) & (image == 0)
opacity = torch.zeros(image.shape)
opacity[background] = 1
opacity[image != 0] += 0.8
opacity[attention_map != 0] += 0.9

combined = torch.stack([image, attention_map, torch.zeros(image.shape), opacity], dim=-1).numpy()
combined_as_uint8 = (combined * 255).astype(np.uint8)

tifffile.imwrite(f"temp_image.tif", combined_as_uint8)
