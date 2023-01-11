import scipy.ndimage as ndi
import numpy as np
from skimage import io
import os
import glob
import tifffile
from tqdm import tqdm

def remove_first_directory(path: str):
    return os.path.join(*path.split(os.path.sep)[1:])

def create_mask(im,intensity_threshold,iteration):
    h,_,_ = ndi.center_of_mass(im)
    h = int(h)
    
    bottom = 0
    for i in range(max(h-(20+iteration),0),0,-1):
        if im[i,:,:].max() < intensity_threshold:
            bottom = i
            break
    
    top = im.shape[0]
    for i in range(min(h+(20+iteration),im.shape[0]),im.shape[0]):
        if im[i,:,:].max() < intensity_threshold:
            top = i
            break

    mask = np.zeros(im.shape)
    mask[bottom:top,:,:] = 1
    
    im_avg = ndi.convolve(im, np.ones((3,3,3))/(3**3))
    
    if im[mask == 1].max() < im_avg.max():
        im[mask == 1] = 0
        return create_mask(im,intensity_threshold,iteration+1)
    return mask

if __name__=="__main__":
    files = glob.glob("sorted_downscaled/**/*.tif")
    for file in tqdm(files):
        im = io.imread(file)
        mask = create_mask(im, 100, 0)
        im[mask == 0] = 0
        new_path = os.path.join("masked", remove_first_directory(file))
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        tifffile.imwrite(new_path, im)
        
    os.makedirs("masked/GH", exist_ok=True)
    os.makedirs("masked/GH", exist_ok=True)