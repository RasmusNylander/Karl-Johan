from torch.nn import functional as F
from tifffile import imwrite, imread
import torch
import numpy as np
import glob
from create_dataloader import make_dataloaders, MNInSecTVariant, MNInSecTVariantMeta, DatasetScale, Augmentation
from tqdm import tqdm
import pandas as pd
import itertools

def mismatch(image,attention,threshold_image=0.25,threshold_att=0.5):
    image[image>threshold_image] = 1
    image[image<=threshold_image] = 0
    
    attention[attention>threshold_att] = 1
    attention[attention<=threshold_att] = 0
    
    if sum(attention[attention==1]) != 0:
        percent_outside = sum(attention[image == 0])/sum(attention[attention==1])
    else:
        percent_outside = 1
    
    return percent_outside

def calculate_outside(layer, scale, network, datavariant, test_loader):
    outside = {"BC":[],"BF":[],"BL":[],"CF":[],"GH":[],"MA":[],"ML":[],"PP":[],"SL":[],"WO":[]}
    for i, (image_batch, label_batch) in enumerate(test_loader):
        image_name = test_loader.dataset.get_name_of_image(i)
        name = test_loader.dataset.label_to_name(label_batch.argmax().item())
        if scale == 0.25:
            scale = "025"
        elif scale == 0.5:
            scale = "050"
        elif scale == 1:
            scale = "100"

        prediction_path = glob.glob(f"attention_maps/{network}_{scale}{dataset_variant.augmentation_suffix}/layer{layer}/{image_name}/*_prediction.tif")
        prediction_map = imread(prediction_path[0])
        prediction_map = torch.from_numpy(prediction_map)
        if scale == "025":
            attention_values = F.interpolate(prediction_map.unsqueeze(dim=0).unsqueeze(dim=0), (64,32,32)).squeeze()
        elif scale == "050":
            attention_values = F.interpolate(prediction_map.unsqueeze(dim=0).unsqueeze(dim=0), (128,64,64)).squeeze()
        elif scale == "100":
            attention_values = F.interpolate(prediction_map.unsqueeze(dim=0).unsqueeze(dim=0), (256,128,128)).squeeze()
            
        percent_outside = mismatch((image_batch[0,0,:,:,:]/image_batch.max()).numpy(),attention_values.numpy())
        outside[name].append(percent_outside)
        
    names = ["BC","BF","BL","CF","MA","ML","PP","SL","WO"]
    means = {}
    
    for n in names:
        means[n] = np.mean(outside[n])
        
    return means
    
if __name__ == "__main__":
    scales = [0.25,0.5,1]
    networks = ["DenseNet121","ResNet18","ResNet50","SEResNet50"]
    augments = ["o","m","t"]
    layer = "1"
    all_dict = {}

    for network in tqdm(networks):
        for scale, augment in itertools.product(scales,augments):
            dataset_variant = MNInSecTVariant(
                    Augmentation.parse_from_string(augment),
                    DatasetScale.from_float(scale)
                )

            _,_,test_loader = make_dataloaders(dataset_variant,batch_size=1)


            temp = calculate_outside(layer, scale, network, dataset_variant, test_loader)

            all_dict[f"{network}_{scale}{dataset_variant.augmentation_suffix}/layer{layer}"] = temp

        pd.DataFrame(all_dict).to_csv(f"stats_{network}.csv")