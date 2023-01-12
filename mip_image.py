import glob
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from tqdm import tqdm
import os

if os.path.exists("./datasets/masked/")
    files = glob.glob("./datasets/masked/**/*.tif")
    for file in tqdm(files):
        im = io.imread(file)
        plt.imsave(file[:-3]+"jpg",np.hstack([np.max(im, axis=1),np.max(im, axis=2)]),cmap="gray")
    
    
files = glob.glob("./datasets/sorted_downscaled/**/*.tif")
for file in tqdm(files):
    im = io.imread(file)
    plt.imsave(file[:-3]+"jpg",np.hstack([np.max(im, axis=1),np.max(im, axis=2)]),cmap="gray")