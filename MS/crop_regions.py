import os
import openslide
import random
import pandas as pd
from tqdm import tqdm


slides_dirs = ["/media/hdd1/neo/BMA_AML", "/media/hdd2/neo/BMA_Normal"]
# save_dir = "/media/hdd3/neo/bma_regions_crop"

save_dir = "/media/hdd3/neo/bma_regions_crop_test"

problem_slides = []

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get all the ndpi slides in the slides_dir
ndpi_slides = []

for slides_dir in slides_dirs:
    for root, dirs, files in os.walk(slides_dir):
        for file in files:
            if file.endswith(".ndpi"):
                ndpi_slides.append(os.path.join(root, file))


print(f"Found {len(ndpi_slides)} slides")


num_per_slide = 1
region_size = 512

current_idx = 55333

metadata = {
    "idx": [],
    "slide_path": [],
    "TL_x": [],
    "TL_y": [],
    "region_size": [],
    "level": [],
    "mpp": [],
}

for slide in tqdm(ndpi_slides, desc="Cropping From Slides"):

    try:
        # open the slide
        slide_obj = openslide.OpenSlide(slide)

        # get the height and width of the slide at level 0
        slide_width, slide_height = slide_obj.dimensions

        # find num_per_slide random regions of dimensions region_size x region_size
        for i in range(num_per_slide):
            # randomly select a TL_x and TL_y
            TL_x = random.randint(0, slide_width - region_size)
            TL_y = random.randint(0, slide_height - region_size)

            # get the region
            region = slide_obj.read_region((TL_x, TL_y), 0, (region_size, region_size))

            # if the region is RGBA, convert it to RGB
            if region.mode == "RGBA":
                region = region.convert("RGB")

            # save the region
            region.save(f"region_{current_idx}.jpg")

            # save the metadata
            metadata["idx"].append(current_idx)
            metadata["slide_path"].append(slide)
            metadata["TL_x"].append(TL_x)
            metadata["TL_y"].append(TL_y)
            metadata["region_size"].append(region_size)
            metadata["level"].append(0)
            metadata["mpp"].append(float(slide_obj.properties["openslide.mpp-x"]))

            current_idx += 1

    except Exception as e:
        raise e

        # add the slide to the problem slides
        problem_slides.append(slide)

        print(f"Error with slide {slide}, skipping...")

    except KeyboardInterrupt:
        raise KeyboardInterrupt


# save the metadata
metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv("crop_metadata.csv", index=False)

# save the problem slides to a text file
with open("problem_slides.txt", "w") as file:
    for slide in problem_slides:
        file.write(f"{slide}\n")
