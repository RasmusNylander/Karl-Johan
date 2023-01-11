import argparse
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

BATCH_SIZE = 1
assert BATCH_SIZE == 1

def save_attention_map(attention_map: Tensor, path: str):
    first_channel = attention_map[0]
    first_channel = first_channel.numpy().transpose(1, 2, 0)
    image_nifti = nib.Nifti1Image(first_channel, affine=np.eye(4))
    nib.save(image_nifti, path)

def generate_attention_maps(
    model_type: ModelType,
    scale: float,
    models_root: str,
    data_path: str,
    device: torch.device,
):
    model_string_id = f"{model_type.name}_{str(int(scale * 100)).zfill(3)}"

    model_path = f"{models_root}/{model_string_id}.pth"

    _, _, test_loader = make_dataloaders(num_workers=0, persistent_workers=False, data_path=data_path, batch_size=BATCH_SIZE, scale=scale)

    dataset: Dataset = test_loader.dataset

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention maps for a model")
    parser.add_argument("--data_path", default="./datasets/sorted_downscaled", type=str)
    parser.add_argument("--models_root", default="./models", type=str)
    parser.add_argument("--output_path", default="./attention_maps", type=str)
    parser.add_argument("--model", type=str, help='ResNet18, ResNet50, DenseNet, SEResNet50', required=True)
    parser.add_argument("--scale", type=float, help='Scale of the images. Must be 0.25, 0.5 or 1.0', required=True)
    parser.add_argument("--cpu", action="store_true", help="Force using CPU")

    args = parser.parse_args()
    data_path = args.data_path
    models_root = args.models_root
    output_path = args.output_path

    scale = args.scale
    assert scale in [0.25, 0.5, 1.0], f"scale of {scale} not yet supported. Scale must be either 0.25, 0.5 or 1.0"

    model_name = args.model
    try:
        model_type = ModelType[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} not supported. Choose from {[type.name for type in ModelType]}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    generate_attention_maps(model_type, scale, models_root, data_path, device)
