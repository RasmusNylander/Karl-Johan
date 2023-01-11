from scipy.ndimage import zoom
import glob
from skimage import io
from tifffile import imwrite
import os
from tqdm import tqdm

scales = [0.25, 0.5]
scale_names = ["64x32x32", "128x64x64"]

image_paths = glob.glob("**[a-zA-Z]/**/*.tif")

for image_path in tqdm(image_paths):
    image = io.imread(image_path)

    for scale, scale_name in zip(scales, scale_names):
        path_components = image_path.split(os.path.sep)
        path_components[0] += "_" + scale_name
        new_path = os.path.join(*path_components)
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)

        scaled_image = zoom(image, scale)
        imwrite(new_path, scaled_image)

    
for dataset in ["sorted_downscaled", "masked"]:
    for scale_name in scale_names:
        directory = f"{dataset}_{scale_name}/GH"
        os.makedirs(directory, exist_ok=True)
    