import os
import sys
import time
import argparse

import torch
from torch import Tensor, device as Device

from create_dataloader import DatasetScale, make_dataloaders
from model_picker import ModelType, get_model
from torchmetrics import ConfusionMatrix
import numpy as np
import tqdm

def create_confusion(model,data_loader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for image_batch, labels in tqdm.tqdm(data_loader,unit="batch"):
            output = model(image_batch.to(device))
            all_labels.append(labels)
            all_outputs.append(output.cpu())
    targets = torch.vstack(all_labels).argmax(dim=1)
    preds = torch.vstack(all_outputs).argmax(dim=1)
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=10)
    cm = confmat(preds,targets)
    return cm
            


if __name__=="__main__":
    MODELS_ROOT = "./models"
    DATA_PATH = "./datasets/MNInSecT"

    for model_type in ModelType:
        for scale in DatasetScale:
            model_path = f"{MODELS_ROOT}/{model_type.name}_{str(int(scale.to_float()*100)).zfill(3)}.pth"

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = get_model(model_type)
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)

            _,_,test_loader = make_dataloaders(
                batch_size=10,
                seed=69420,
                data_path=DATA_PATH,
                pin_memory=False,
                as_rgb=False,
                scale=scale,
                masked = True
            )
    
            confmat = create_confusion(model,test_loader)

            np.save(f"confusion_matrix/{model_type.name}_{str(int(scale.to_float()*100)).zfill(3)}",confmat.numpy())