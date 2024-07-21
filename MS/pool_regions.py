import os
import pandas as pd
from tqdm import tqdm


data_dir = "/media/hdd3/neo/results_bma_v4"
save_dir = "/media/hdd3/neo/PL1_cell_scan_training_data_non_pl1_v2_dup"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

symbolic = True

# get all the subfolders that do not start with ERROR
subfolders = [f for f in os.listdir(data_dir) if not f.startswith("ERROR")]
# make sure to check if they are actually folders, not files
subfolders = [f for f in subfolders if os.path.isdir(os.path.join(data_dir, f))]

current_index = 0

metadata = {
    "idx": [],
    "original_path": [],
    "symbolic": [],
}

for folder in tqdm(subfolders, desc="Processing Subfolders"):
    # copy all the images in folder/focus_regions/high_mag_unannotated to the save_dir as symbolic links if symbolic is True
    # or as actual files if symbolic is False
    # the new file names will be the current_index dot jpg

    folder_path = os.path.join(data_dir, folder)
    focus_regions_path = os.path.join(folder_path, "focus_regions")

    if not os.path.exists(focus_regions_path):
        continue

    high_mag_unannotated_path = os.path.join(focus_regions_path, "high_mag_unannotated")

    if not os.path.exists(high_mag_unannotated_path):
        continue

    image_files = [
        f
        for f in os.listdir(high_mag_unannotated_path)
        if f.endswith(".jpg")
        or f.endswith(".png")
        or f.endswith(".jpeg")
        or f.endswith(".tif")
    ]

    for image_file in image_files:
        image_path = os.path.join(high_mag_unannotated_path, image_file)
        new_image_path = os.path.join(save_dir, f"{current_index}.jpg")

        if symbolic:
            os.symlink(image_path, new_image_path)
        else:
            os.copy(image_path, new_image_path)

        metadata["idx"].append(current_index)
        metadata["original_path"].append(image_path)
        metadata["symbolic"].append(symbolic)

        current_index += 1

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(os.path.join(save_dir, "metadata_region_pool.csv"), index=False)
print("Data compiled successfully.")
