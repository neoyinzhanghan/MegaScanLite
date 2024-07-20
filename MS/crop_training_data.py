import os
import pandas as pd
import openslide
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

cellname = "PL1"

cell_data_path = "/media/ssd2/clinical_text_data/MegakaryoctePltClumpProject/slides_with_pl_cells_renamed.csv"
slide_folder = "/dmpisilon_tools/Greg/SF_Data/Pathology Images"
save_dir = "/media/hdd3/neo/PL1_cell_scan_training_data_v2"
desired_mpp = 0.2297952524300848  # Microns per pixel
num_regions_per_cell = 10
region_size = 512  # Region size at the desired_mpp

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(cell_data_path)

metadata = {
    "data_idx": [],
    "slide_path": [],
    "center_x": [],
    "center_y": [],
    "cellname": [],
    "cell_image_size": [],
    "region_TL_x": [],
    "region_TL_y": [],
    "region_BR_x": [],
    "region_BR_y": [],
    "region_size": [],
    "center_x_rel": [],
    "center_y_rel": [],
}

current_index = 0
problem_slides = []


def find_file_recursive(slide_folder, slide_name):
    slide_folder_path = Path(slide_folder)
    for file_path in slide_folder_path.rglob(slide_name):
        return file_path
    return None


for i, row in tqdm(df.iterrows(), desc="Processing Cell Instances", total=len(df)):
    if row["cell_type"] != cellname:
        continue
    slide_name = row["slide_name"]
    slide_path = find_file_recursive(slide_folder, slide_name)
    if slide_path is None:
        problem_slides.append(slide_name)
        continue

    center_x = int(row["center_x_slide"])
    center_y = int(row["center_y_slide"])
    cell_image_size = 256  # Constant in pixels at the highest resolution

    try:
        slide = openslide.OpenSlide(str(slide_path))
        best_level = slide.get_best_level_for_downsample(
            desired_mpp / float(slide.properties["openslide.mpp-x"]) / desired_mpp
        )

        scale_factor_best_level = slide.level_downsamples[best_level]
        best_level_mpp = (
            float(slide.properties["openslide.mpp-x"]) * scale_factor_best_level
        )

        downsample_factor_from_best_level = desired_mpp / best_level_mpp

        # Convert the coordinates and sizes to the best level
        scaled_best_level_center_x = int(center_x / scale_factor_best_level)
        scaled_best_level_center_y = int(center_y / scale_factor_best_level)
        scaled_best_level_region_size = int(
            region_size * downsample_factor_from_best_level
        )
        scaled_best_level_cell_image_size = int(
            cell_image_size * downsample_factor_from_best_level
        )

        # Calculate region
        min_TL_x = (
            scaled_best_level_center_x
            - (scaled_best_level_region_size - scaled_best_level_cell_image_size // 2)
            - scaled_best_level_cell_image_size // 2
        )
        max_TL_x = (
            scaled_best_level_center_x
            - scaled_best_level_cell_image_size // 2
            + scaled_best_level_cell_image_size // 2
        )
        min_TL_y = (
            scaled_best_level_center_y
            - (scaled_best_level_region_size - scaled_best_level_cell_image_size // 2)
            - scaled_best_level_cell_image_size // 2
        )
        max_TL_y = (
            scaled_best_level_center_y
            - scaled_best_level_cell_image_size // 2
            + scaled_best_level_cell_image_size // 2
        )  # the rationale here is that we allow less than or equal to half of the cells to be cropped out

        for _ in range(num_regions_per_cell):
            best_level_region_TL_x = random.randint(min_TL_x, max_TL_x)
            best_level_region_TL_y = random.randint(min_TL_y, max_TL_y)
            region = slide.read_region(
                (best_level_region_TL_x, best_level_region_TL_y),
                best_level,
                (scaled_best_level_region_size, scaled_best_level_region_size),
            )
            if region.mode == "RGBA":
                region = region.convert("RGB")

            region = region.resize(
                (
                    region.size[0] // downsample_factor_from_best_level,
                    region.size[1] // downsample_factor_from_best_level,
                ),
                Image.Resampling.LANCZOS,
            )

            assert region.size == (
                region_size,
                region_size,
            ), f"Region size is {region.size}, which is not equal to the desired region size {region_size}"
            region.save(os.path.join(save_dir, f"{current_index}.jpg"))

            metadata["data_idx"].append(current_index)
            metadata["slide_path"].append(str(slide_path))
            metadata["center_x"].append(center_x)
            metadata["center_y"].append(center_y)
            metadata["cellname"].append(cellname)
            metadata["cell_image_size"].append(cell_image_size)
            metadata["region_TL_x"].append(
                best_level_region_TL_x * scale_factor_best_level
            )
            metadata["region_TL_y"].append(
                best_level_region_TL_y * scale_factor_best_level
            )
            metadata["region_BR_x"].append(
                (best_level_region_TL_x + scaled_best_level_region_size)
                * scale_factor_best_level
            )
            metadata["region_BR_y"].append(
                (best_level_region_TL_y + scaled_best_level_region_size)
                * scale_factor_best_level
            )
            metadata["region_size"].append(region_size)
            metadata["center_x_rel"].append(
                (scaled_best_level_center_x - best_level_region_TL_x)
                / scaled_best_level_region_size
            )
            metadata["center_y_rel"].append(
                (center_y - best_level_region_TL_y * scale_factor_best_level)
                / scaled_best_level_region_size
            )
            current_index += 1

        slide.close()

    except Exception as e:
        print(f"Problem with slide {slide_path}: {e}")
        problem_slides.append(str(slide_path))
        continue

    # if keyboard interrupt then just end the loop
    except KeyboardInterrupt:
        raise KeyboardInterrupt

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

with open(os.path.join(save_dir, "problem_slides.txt"), "w") as f:
    for slide in problem_slides:
        f.write(f"{slide}\n")
