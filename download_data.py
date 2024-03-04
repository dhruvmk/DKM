import os
import zipfile

from huggingface_hub import hf_hub_download
'''
invalid = [7, 11, 14, 22, 30, 32, 39]
# Zip files of segments to extract
zipfiles = {}
for i in range(64):
    if i in invalid:
        continue
    number = str(i).zfill(8)
    zipfiles["clusters/train/dawn_vesta/" + number + ".zip"] = number

print(zipfiles)

index = 0
for file in zipfiles:
    hf_hub_download(
        repo_id="travisdriver/astrovision-data",
        filename=file,
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False
    )

    if index == 0 or index == 1:
        with zipfile.ZipFile("data/" + file, "r") as zip_ref:
            zip_ref.extractall("data/")
    else:
        with zipfile.ZipFile("data/" + file, "r") as zip_ref:
            zip_ref.extractall("data/" + zipfiles[file])
    index += 1
'''

with zipfile.ZipFile("data/clusters/train/dawn_vesta/00000005.zip", "r") as zip_ref:
    zip_ref.extractall("data/")
