import argparse
import copy

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
        scale: DatasetScale,
        models_root: str,
        data_path: str,
        device: torch.device,
        layer: int,
        dataset_augmentation: Augmentation,
        leave_progress_bar=True
):
    dataset_variant = MNInSecTVariant(dataset_augmentation, scale)

    model_string_id = get_model_name(model_type, dataset_variant)

    model_path = f"{models_root}/{model_string_id}.pth"

    _, _, test_loader = make_dataloaders(num_workers=0, persistent_workers=False, data_path=data_path,
                                         batch_size=BATCH_SIZE, pin_memory=False, variant=dataset_variant)

    dataset: Dataset = test_loader.dataset

    model = get_pretrained(model_type, dataset_variant, models_root).to(device)
    model.eval()

    layer_name = layer_to_layer_name(model_type, layer)
    models = create_injected_models(model, dataset.num_classes(), layer_name)
    model_best = models.pop(0)

    image_output_root = f"attention_maps/{model_string_id}/layer{layer}"
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
    parser.add_argument("--dataset_variants", type=str, nargs="+", default="original", help="Variant of MNInSecT to use. Must be 'original' or 'masked'", required=True)

    args = parser.parse_args()
    data_path: str = args.data_path
    models_root: str = args.models_root
    output_path: str = args.output_path
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    dataset_augmentations = [Augmentation.parse_from_string(string) for string in args.dataset_variants]

    for scale in args.scales:
        assert scale in [0.25, 0.5, 1.0], f"scale of {scale} not yet supported. Scale must be either 0.25, 0.5 or 1.0"
    scales = [DatasetScale.from_float(scale) for scale in args.scales]

    model_names = args.models
    model_types: list[ModelType] = []
    for model_name in model_names:
        try:
            model_types.append(ModelType[model_name])
        except KeyError:
            raise ValueError(f"Model {model_name} not supported. Choose from {[type.name for type in ModelType]}")

    layers: list[int] = args.layers
    for layer in layers:
        assert layer in [1, 2, 3, 4], f"Layer {layer} not yet supported. Layer must be either 1, 2, 3 or 4"

    for model_type, dataset_augmentation, scale, layer in itertools.product(model_types, dataset_augmentations, scales, layers, desc="Generating attention maps", unit="model"):
        generate_attention_maps(model_type, scale, models_root, data_path, device, layer, dataset_augmentation, False)
