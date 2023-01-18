from enum import Enum, auto
import torch
from monai.networks.nets import densenet121, SEResNet50, ResNet

from create_dataloader import MNInSecTVariant


class ModelType(Enum):
    ResNet18 = auto(),
    ResNet50 = auto(),
    DenseNet121 = auto(),
    SEResNet50 = auto(),

    @staticmethod
    def parse_from_string(string: str):
        assert len(ModelType) == 4
        match string.lower():
            case "resnet18":
                return ModelType.ResNet18
            case "resnet50":
                return ModelType.ResNet50
            case "densenet121":
                return ModelType.DenseNet121
            case "seresnet50":
                return ModelType.SEResNet50
            case _:
                raise ValueError(f"Unknown model: {string}. Valid models are: {[model.name for model in ModelType]}")

    def next(self):
        assert len(ModelType) == 4
        if self == ModelType.ResNet18:
            return ModelType.ResNet50
        elif self == ModelType.ResNet50:
            return ModelType.DenseNet121
        elif self == ModelType.DenseNet121:
            return ModelType.SEResNet50
        elif self == ModelType.SEResNet50:
            return ModelType.ResNet18

    def previous(self):
        assert len(ModelType) == 4
        if self == ModelType.ResNet18:
            return ModelType.SEResNet50
        elif self == ModelType.ResNet50:
            return ModelType.ResNet18
        elif self == ModelType.DenseNet121:
            return ModelType.ResNet50
        elif self == ModelType.SEResNet50:
            return ModelType.DenseNet121

    def create(self) -> torch.nn.Module:
        match self:
            case ModelType.ResNet18:
                return ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[32, 64, 128, 256],
                              num_classes=10, n_input_channels=1)
            case ModelType.ResNet50:
                return ResNet(block="bottleneck", layers=[3, 4, 6, 3], block_inplanes=[32, 64, 128, 256],
                              num_classes=10, n_input_channels=1)
            case ModelType.DenseNet121:
                return densenet121(spatial_dims=3, in_channels=1, out_channels=10)
            case ModelType.SEResNet50:
                return SEResNet50(spatial_dims=3, in_channels=1, out_channels=10)



def get_model_name(type: ModelType, variant: MNInSecTVariant) -> str:
    scale_suffix = f"_{str(variant.scale).zfill(3)}"
    return f"{type.name}{scale_suffix}{variant.augmentation_suffix}"


def get_pretrained(type: ModelType, variant: MNInSecTVariant, models_root: str, map_location=None) -> torch.nn.Module:
    name = get_model_name(type, variant)
    model_path = f"{models_root}/{name}.pth"
    model = type.create()
    model.load_state_dict(torch.load(model_path, map_location=map_location)['net'], strict=True)
    return model
