from scipy.ndimage import zoom
import glob
from skimage import io
from tifffile import imwrite
import os

image_paths = glob.glob("**/**/*.tif")

for image_path in image_paths:
    image = io.imread(image_path)
    im_05 = zoom(image,0.5)
    im_025 = zoom(image,0.25)
    
    path_05 = image_path.split("/")
    path_05[0] = path_05[0]+"_128x64x64"
    path_05 = "/".join(path_05)
    path_025 = image_path.split("/")
    path_025[0] = path_025[0]+"_64x32x32"
    path_025 = "/".join(path_025)
    
    os.makedirs("/".join(path_05.split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(path_025.split("/")[:-1]), exist_ok=True)
    
    
    imwrite(path_05,im_05)
    imwrite(path_025,im_025)
    
os.makedirs("sorted_downscaled_128x64x64/GH", exist_ok=True)
os.makedirs("sorted_downscaled_64x32x32/GH", exist_ok=True)
    