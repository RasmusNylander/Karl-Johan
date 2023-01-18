import torch
from tqdm.contrib import itertools


from create_dataloader import DatasetScale, Augmentation, MNInSecTVariant, make_dataloaders
from model_picker import ModelType, get_model_name, get_pretrained
from torchmetrics import ConfusionMatrix
import numpy as np
import tqdm
import os
import glob

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
    return cm, targets, predictions
            


if __name__=="__main__":
    MODELS_ROOT = "./models"
    DATA_PATH = "./datasets/MNInSecT"

    for model_type, dataset_variant in itertools.product(ModelType, MNInSecTVariant):
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

        confmat,targets,predictions = create_confusion(model, test_loader)
        
        path_to_attentionmaps = [f"attention_maps/{model_name}/layer1/{os.path.split(p)[-1][:-4]}/" for p in test_loader.dataset.image_paths]
        
        for pred, path in zip(predictions,path_to_attentionmaps):
            pred_class = test_loader.dataset.label_to_name(pred.item())
            prev_pred = os.path.split(glob.glob(f'{path}*_prediction.tif')[0])[-1]
            
            if pred_class not in prev_pred:
                os.rename(f'{path}{pred_class}.tif',f'{path}{pred_class}_prediction.tif')
                os.rename(f'{path}{prev_pred}',f'{path}{pred_class[:-15]}.tif')
                
        
        np.save(f"confusion_matrix/{model_name}", confmat.numpy())