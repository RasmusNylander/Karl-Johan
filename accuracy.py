from torch import Tensor


def accuracy(predictions: Tensor, labels: Tensor) -> float:
    return (
        predictions.argmax(dim=1) == labels.argmax(dim=1)
    ).sum().item() / labels.shape[0]