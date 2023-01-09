import os
import shutil

data_path = "sorted_downscaled"

old_file_names = ['cf_002_downscaled.tif',
                  'cf_003_downscaled.tif','cf_004_downscaled.tif',
                  'cf_005_downscaled.tif','cf_006_downscaled.tif',
                  'cf_007_downscaled.tif','cf_112_downscaled.tif',
                  'cf_113_downscaled.tif','cf_114_downscaled.tif',
                  'cf_115_downscaled.tif']

if len(os.listdir(data_path+"/CF/")) == 119:
    n = len(os.listdir(data_path+"/BC/"))
    for i,file in enumerate(old_file_names):
        shutil.move(f"{data_path}/CF/{file}", f"{data_path}/BC/bc_{n+i:03}_downscaled.tif")


    cf = os.listdir(data_path+"/CF/")
    cf.sort()

    for i,file in enumerate(cf):
        os.rename(f"{data_path}/CF/{file}",f"{data_path}/CF/cf_{i:03}_downscaled.tif")
    