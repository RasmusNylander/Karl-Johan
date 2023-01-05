import os
import cv2
import glob
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms


class dataset (Dataset):
    def __init__(self,
                 train,
                 transform=True,
                 seed=42,
                 data_path='../datasets/sorted_downscaled'):
        self.transform = transform
        
        # data_path = os.path.join(data_path, 'train' if train else 'test')
        
        self.image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        self.image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(self.image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.tif')
        self.rng = np.random.default_rng(seed=seed)
    
    def __len__(self):
        return len(self.image_paths) #len(self.data)
    
    def __getitem__(self,idx):        
        image_path = self.image_paths[idx]
        
        image = io.imread(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        
        #image = cv2.resize(image,self.size,interpolation=cv2.INTER_LINEAR)
        if self.transform:
            flips = lambda x: [np.fliplr(x),
                               np.flipud(x),
                               np.flipud(np.fliplr(x)),
                               np.rot90(x,k=self.rng.choice([1,2,3]),axes=(1,2)),
                               x]
            image = self.rng.choice(flips(image))
        
        X = transforms.functional.to_tensor(image)
        
        
        return X,y
    
    def get_image_paths(self):
        return self.image_paths
    
    def get_image_classes(self):
        return self.image_classes



def make_dataloaders(batch_size=16,transform=True,seed=42):
    """
    Creates a train and test dataloader with a variable batch size and image shape.
    And using a weighted sampler for the training dataloader to have balanced mini-batches when training.
    """
    train_set = dataset(train=True,transform=transform, seed=seed)
    test_set = dataset(train=False,transform=False, seed=seed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, worker_init_fn=np.random.seed(seed),num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(seed),num_workers=4, pin_memory=True)    
    
    return train_loader,test_loader
