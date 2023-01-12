import os

DATASET_PATH = "./MNInSecT/256x128x128"

folders = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]
files = [f"{folder}/{file}" for folder in folders for file in os.listdir(f"{DATASET_PATH}/{folder}")]
# write files as csv
with open(f"{DATASET_PATH}/files.csv", "w") as f:
    f.write("files\n")
    f.write("\n".join(files))
