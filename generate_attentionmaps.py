import copy
import os

import numpy as np
import torch
from monai.data.image_reader import nib
from torch import Tensor
from tqdm import tqdm

from create_dataloader import Dataset, make_dataloaders
from medcam import medcam
from model_picker import ModelType, get_model

model_type: ModelType = ModelType.ResNet18
scale: float = 0.5
assert scale in [0.25, 0.5, 1.0]


MODELS_ROOT = "./models"
DATA_PATH = "./datasets/sorted_downscaled"
BATCH_SIZE = 1
assert BATCH_SIZE == 1

def save_attention_map(attention_map: Tensor, path: str):
    first_channel = attention_map[0]
    first_channel = first_channel.numpy().transpose(1, 2, 0)
    image_nifti = nib.Nifti1Image(first_channel, affine=np.eye(4))
    nib.save(image_nifti, path)

model_string_id = f"{model_type.name}_{str(int(scale * 100)).zfill(3)}"

model_path = f"{MODELS_ROOT}/{model_string_id}.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
_, _, test_loader = make_dataloaders(num_workers=0, persistent_workers=False, data_path=DATA_PATH,
                                     batch_size=BATCH_SIZE, scale=scale)
dataset: Dataset = test_loader.dataset
num_classes = len(dataset.get_image_classes())

model = get_model(model_type)
model.eval()
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)

models = [copy.copy(model)]
medcam.inject(models[0], return_attention=True, layer="auto", label="best")
for label in range(dataset.num_classes()):
    medcam_model = copy.copy(model)
    medcam_model = medcam.inject(medcam_model, return_attention=True, layer="auto", label=label)
    models.append(medcam_model)

image_output_root = f"attention_maps/{model_string_id}/layer"
assert BATCH_SIZE == 1
for image_id, (image_batch, batch_labels) in enumerate(tqdm(test_loader, unit="image")):
    image_name = dataset.get_name_of_image(image_id)
    image_dir = f"{image_output_root}/{image_name}"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_batch = image_batch.to(device)

    prediction, attention_map_predicted_label = models[0](image_batch)
    prediction_label = prediction[0].argmax(dim=0).item()
    save_attention_map(attention_map_predicted_label[0].detach().cpu(),
                       f"{image_dir}/{dataset.label_to_name(prediction_label)}")

    correct_label = batch_labels[0].argmax(dim=0).item()
    if correct_label == prediction_label:
        continue
    _, attention_map_correct_label = models[correct_label + 1](image_batch)
    save_attention_map(attention_map_correct_label[0].detach().cpu(),
                       f"{image_dir}/{dataset.label_to_name(correct_label)}")
