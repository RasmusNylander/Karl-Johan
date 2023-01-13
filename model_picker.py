from enum import Enum, auto
import torch
from monai.networks.nets import densenet121, SEResNet50, ResNet

from create_dataloader import DatasetScale, Augmentation


class ModelType(Enum):
    ResNet18 = auto(),
    ResNet50 = auto(),
    DenseNet121 = auto(),
    SEResNet50 = auto(),

def get_model(type: ModelType) -> torch.nn.Module:
    match type:
        case ModelType.ResNet18:
            return ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[32,64,128,256], num_classes=10, n_input_channels=1)
        case ModelType.ResNet50:
            return ResNet(block="bottleneck", layers=[3, 4, 6, 3], block_inplanes=[32,64,128,256], num_classes=10, n_input_channels=1)
        case ModelType.DenseNet121:
            return densenet121(spatial_dims=3, in_channels=1, out_channels=10)
        case ModelType.SEResNet50:
            return SEResNet50(spatial_dims=3, in_channels=1, num_classes=10)


def get_model_name(type: ModelType, dataset_variant: Augmentation, scale: DatasetScale) -> str:
    assert len(ModelType) == 4 and len(DatasetScale) == 3

    scale_as_string = str(scale).zfill(3)
    scale_suffix = f"_{scale_as_string}"

    assert len(Augmentation) == 3
    if dataset_variant == Augmentation.Original:
        variant_suffix = ""
    elif (dataset_variant == Augmentation.Masked):
        variant_suffix = "_masked"
    elif (dataset_variant == Augmentation.Threshold):
        variant_suffix = "_threshold"

    return f"{type.name}{scale_suffix}{variant_suffix}"

def get_pretrained(type: ModelType, dataset_variant: Augmentation, scale: DatasetScale, models_root: str, map_location=None) -> torch.nn.Module:
    name = get_model_name(type, dataset_variant, scale)
    model_path = f"{models_root}/{name}.pth"
    model = get_model(type)
    model.load_state_dict(torch.load(model_path, map_location=map_location)['net'], strict=True)
    return model