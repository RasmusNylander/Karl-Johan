import torch
from tqdm.contrib import itertools


from create_dataloader import DatasetScale, Augmentation, MNInSecTVariant, make_dataloaders
from model_picker import ModelType, get_model, get_model_name, get_pretrained
from torchmetrics import ConfusionMatrix
import numpy as np
import tqdm

def create_confusion(model, data_loader):
    images_processed = 0
    targets = torch.empty(len(data_loader.dataset))
    predictions = torch.empty(len(data_loader.dataset))
    model.eval()
    with torch.no_grad():
        for image_batch, labels in tqdm.tqdm(data_loader, unit="batch", leave=False):
            output = model(image_batch.to(device))
            targets[images_processed:images_processed+len(image_batch)] = labels.argmax(dim=1)
            predictions[images_processed:images_processed+len(image_batch)] = output.argmax(dim=1).cpu()
            images_processed += len(image_batch)

    
    confmat = ConfusionMatrix(task="multiclass", num_classes=10)
    cm = confmat(predictions, targets)
    return cm
            


if __name__=="__main__":
    MODELS_ROOT = "./models"
    DATA_PATH = "./datasets/MNInSecT"

    for model_type, scale, augmentation in itertools.product(ModelType, DatasetScale, Augmentation):
        dataset_variant = MNInSecTVariant(augmentation, scale)
        model_name = get_model_name(model_type, dataset_variant)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_pretrained(model_type, dataset_variant, MODELS_ROOT, map_location=device).to(device)

        _,_,test_loader = make_dataloaders(
            batch_size=10,
            seed=69420,
            data_path=DATA_PATH,
            pin_memory=False,
            as_rgb=False,
            variant=dataset_variant
        )

        confmat = create_confusion(model, test_loader)

        np.save(f"confusion_matrix/{model_name}", confmat.numpy())