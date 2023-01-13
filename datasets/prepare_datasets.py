import os
import glob
import shutil
import zipfile
import numpy as np
from tqdm import tqdm
from skimage import io
import scipy.ndimage as ndi
from tifffile import imwrite
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def remove_first_directory(path: str):
    return os.path.join(*path.split(os.path.sep)[2:])

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

def move_wrongly_labelled_images(data_path):
    old_file_names = ['cf_002_downscaled.tif',
                      'cf_003_downscaled.tif','cf_004_downscaled.tif',
                      'cf_005_downscaled.tif','cf_006_downscaled.tif',
                      'cf_007_downscaled.tif','cf_112_downscaled.tif',
                      'cf_113_downscaled.tif','cf_114_downscaled.tif',
                      'cf_115_downscaled.tif']

    n = len(os.listdir(data_path+"/BC/"))
    for i,file in enumerate(old_file_names):
        shutil.move(f"{data_path}/CF/{file}", f"{data_path}/BC/bc_{n+i:03}_downscaled.tif")

    cf = os.listdir(data_path+"/CF/")
    cf.sort()

    for i,file in enumerate(cf):
        os.rename(f"{data_path}/CF/{file}",f"{data_path}/CF/cf_{i:03}_downscaled.tif")
        
def create_masked_images():
    files = glob.glob("MNInSecT_new/256x128x128/**/*.tif")
    for file in tqdm(files):
        im = io.imread(file)
        mask = create_mask(im, 100, 0)
        im[mask == 0] = 0
        new_path = os.path.join("MNInSecT_new/256x128x128_masked", remove_first_directory(file))
        imwrite(new_path, im)
        im[im < 25] = 0
        new_path = os.path.join("MNInSecT_new/256x128x128_threshold", remove_first_directory(file))
        imwrite(new_path, im)
        
def scale_and_save_image(image_path, scales, scale_names):
    image = io.imread(image_path)

    for scale, scale_name in zip(scales, scale_names):
        path_components = image_path.split(os.path.sep)
        if "masked" in path_components[1]:
            path_components[1] = f'{scale_name}_masked'
        elif "threshold" in path_components[1]:
            path_components[1] = f'{scale_name}_threshold'
        else:
            path_components[1] = scale_name
        new_path = os.path.join(*path_components)

        scaled_image = ndi.zoom(image, scale)
        imwrite(new_path, scaled_image)
        
if __name__=="__main__":
    #Extract files
    with zipfile.ZipFile("sorted_downscaled.zip", 'r') as zip_ref:
        zip_ref.extractall("MNInSecT_new/")
    
    #Rename base folder
    os.rename("MNInSecT_new/sorted_downscaled","MNInSecT_new/256x128x128")
    
    #Create folder structure
    class_folders = os.listdir("MNInSecT_new/256x128x128/")
    for folder in class_folders:
        os.makedirs(f"MNInSecT_new/128x64x64/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/64x32x32/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/256x128x128_masked/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/128x64x64_masked/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/64x32x32_masked/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/256x128x128_threshold/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/128x64x64_threshold/{folder}",exist_ok=True)
        os.makedirs(f"MNInSecT_new/64x32x32_threshold/{folder}",exist_ok=True)
    
    #Moves wrongly labelled images and renames files
    move_wrongly_labelled_images("MNInSecT_new/256x128x128")
    
    #Creates and saves masked images
    print("\nCreating masked images\n")
    create_masked_images()
    
    #Scales both original and masked images
    original_prefix = "256x128x128"
    scales = [0.25, 0.5]
    scale_prefixes = ["64x32x32", "128x64x64"]

    image_paths = glob.glob(f"MNInSecT_new/{original_prefix}*/**/*.tif")
    print("\nDownscaling images\n")
    Parallel(n_jobs=16)(delayed(scale_and_save_image)(image_path, scales, scale_prefixes) for image_path in tqdm(image_paths, unit="image", desc="Scaling images"))
    
    print('\ncreating maximum intensity projections\n')
    for image_path in tqdm(image_paths):
        im = io.imread(image_path)
        plt.imsave(image_path[:-3]+"jpg",np.hstack([np.max(im, axis=1),np.max(im, axis=2)]),cmap="gray")