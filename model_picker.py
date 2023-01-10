from enum import Enum, auto
import torch
from monai.networks.nets import densenet121, SEResNet50, ResNet


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