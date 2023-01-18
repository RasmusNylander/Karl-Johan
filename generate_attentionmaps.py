import argparse
import copy
from itertools import product

from tifffile import tifffile
from tqdm.contrib import itertools
import os

import torch
from torch import Tensor
from tqdm import tqdm

from create_dataloader import Dataset, DatasetScale, Augmentation, MNInSecTVariant, make_dataloaders
from medcam import medcam
from model_picker import ModelType, get_model, get_pretrained, get_model_name

BATCH_SIZE = 1
assert BATCH_SIZE == 1


def save_attention_map(attention_map: Tensor, path: str):
    first_channel = attention_map[0]
    tifffile.imwrite(f"{path}.tif", first_channel.numpy())

def create_injected_models(base_model: torch.nn.Module, num_classes: int, layer_name: str) -> list[torch.nn.Module]:
    models = [copy.copy(base_model)]
    medcam.inject(models[0], return_attention=True, layer=layer_name, label="best")
    for label in range(num_classes):
        medcam_model = copy.copy(base_model)
        medcam_model = medcam.inject(medcam_model, replace=True, layer=layer_name, label=label)
        models.append(medcam_model)
    return models


def layer_to_layer_name(model: ModelType, layer: int) -> str:
    match model:
        case ModelType.ResNet18:
            return f"layer{layer}"
        case ModelType.ResNet50:
            return f"layer{layer}"
        case ModelType.DenseNet121:
            return f"features.denseblock{layer}"
        case ModelType.SEResNet50:
            return f"layer{layer}"


def generate_attention_maps(
        model_type: ModelType,
        model_dataset_variant: MNInSecTVariant | None,
        dataset_variant: MNInSecTVariant,
        models_root: str,
        data_path: str,
        device: torch.device,
        layer: int,
        leave_progress_bar=True
):
    if model_dataset_variant is None:
        model_dataset_variant = dataset_variant
    model_string_id = get_model_name(model_type, model_dataset_variant)

    _, _, test_loader = make_dataloaders(num_workers=0, persistent_workers=False, data_path=data_path,
                                         batch_size=BATCH_SIZE, pin_memory=False, variant=dataset_variant)

    dataset: Dataset = test_loader.dataset

    model = get_pretrained(model_type, model_dataset_variant, models_root).to(device)
    model.eval()

    layer_name = layer_to_layer_name(model_type, layer)
    models = create_injected_models(model, dataset.num_classes(), layer_name)
    model_best = models.pop(0)

    if model_dataset_variant == dataset_variant:
        image_output_root = f"attention_maps/{model_string_id}/layer{layer}"
    else:
        image_output_root = f"attention_maps/{model_string_id}_{dataset_variant.name}/layer{layer}"
    assert BATCH_SIZE == 1
    for image_id, (image_batch, batch_labels) in enumerate(tqdm(test_loader, unit="image", leave=leave_progress_bar, desc=f"{model_string_id}, layer {layer}")):
        image_name = dataset.get_name_of_image(image_id)
        image_dir = f"{image_output_root}/{image_name}"
        os.makedirs(image_dir, exist_ok=True)

        image_batch = image_batch.to(device)

        correct_label = batch_labels[0].argmax(dim=0).item()
        prediction, attention_map = model_best(image_batch)
        attention_map = attention_map.detach()[0].cpu()
        model_best.zero_grad()
        prediction_label = prediction[0].argmax(dim=0).item()
        map_name = f"{dataset.label_to_name(prediction_label)}_prediction"
        save_attention_map(attention_map, f"{image_dir}/{map_name}")

        for label, model in enumerate(models):
            if label == prediction_label:
                continue

            with torch.no_grad():
                attention_map = model(image_batch)
                attention_map = attention_map.detach()[0].cpu()
                map_name = f"{dataset.label_to_name(label)}"
                save_attention_map(attention_map, f"{image_dir}/{map_name}")
                model.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention maps for a model")
    parser.add_argument("--data_path", default="./datasets/MNInSecT", type=str)
    parser.add_argument("--models_root", default="./models", type=str)
    parser.add_argument("--output_path", default="./attention_maps", type=str)
    parser.add_argument("--models", type=str, nargs="+", help='ResNet18, ResNet50, DenseNet121, SEResNet50')
    parser.add_argument("--scales", type=float, nargs="+", help='Scale of the images. Must be 0.25, 0.5 or 1.0')
    parser.add_argument("--layers", type=int, nargs="+", help='Layer to use for attention maps.')
    parser.add_argument("--cpu", action="store_true", help="Force using CPU")
    parser.add_argument("--model_dataset_augmentations", type=str, default="same")
    parser.add_argument("--dataset_augmentations", type=str, nargs="+", default="original")

    args = parser.parse_args()
    data_path: str = args.data_path
    models_root: str = args.models_root
    output_path: str = args.output_path
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    dataset_augmentations = [Augmentation.parse_from_string(string) for string in args.dataset_augmentations]
    scales = [DatasetScale.from_float(scale) for scale in args.scales]
    dataset_variants = [
        MNInSecTVariant(augmentation, scale) for augmentation, scale in product(dataset_augmentations, scales)
    ]

    model_types = [ModelType.parse_from_string(model) for model in args.models]

    if args.model_dataset_augmentations == "same":
        model_dataset_augmentations = None
    else:
        model_dataset_augmentations = Augmentation.parse_from_string(args.model_dataset_augmentations)

    layers: list[int] = args.layers
    for layer in layers:
        assert layer in [1, 2, 3, 4], f"Layer {layer} not yet supported. Layer must be either 1, 2, 3 or 4"

    for model_type, dataset_variant, layer in itertools.product(model_types, dataset_variants, layers, desc="Generating attention maps", unit="model"):
        model_dataset_variant = MNInSecTVariant(model_dataset_augmentations, dataset_variant.scale) if model_dataset_augmentations is not None else None
        generate_attention_maps(model_type, model_dataset_variant, dataset_variant, models_root, data_path, device, layer, False)
