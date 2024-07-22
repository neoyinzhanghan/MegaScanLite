import numpy as np
import ray
import openslide
from MSFocuRegion import FocusRegion
from utils import crop_image_at_mpp


@ray.remote
class WSICropper:
    """=== Class Attributes ===
    - level
    - wsi_path
    """

    def __init__(self, wsi_path, mpp):

        self.wsi_path = wsi_path
        self.mpp = mpp

    def async_crop_focus_regions_batch(self, focus_regions, mpp):

        # traverse through a list of focus regions

        wsi = openslide.OpenSlide(self.wsi_path)

        for focus_region in focus_regions:
            level_0_coord = focus_region.level_0_coord

            level_0_height = focus_region.level_0_height
            level_0_width = focus_region.level_0_widthsss

            # use the crop_image_at_mpp function to crop the image

            level_0_mpp = focus_region.level_0_mpp

            downsample_factor = level_0_mpp / mpp

            # if the level_0_height or the level_0_width is not divisible by the downsample factor, raise an error
            assert (
                level_0_height % downsample_factor == 0
            ), f"Height {level_0_height} is not divisible by the downsample factor {downsample_factor}, when cropping focus region at {level_0_coord} with height {level_0_height} and width {level_0_width} at {level_0_mpp} mpp"
            assert (
                level_0_width % downsample_factor == 0
            ), f"Width {level_0_width} is not divisible by the downsample factor {downsample_factor}, when cropping focus region at {level_0_coord} with height {level_0_height} and width {level_0_width} at {level_0_mpp} mpp"

            height = level_0_height // downsample_factor
            width = level_0_width // downsample_factor

            image_crop = crop_image_at_mpp(
                level_0_coord, height=height, width=width, mpp=mpp, wsi=wsi
            )

            focus_region.get_image(mpp, image_crop)

        # close the wsi
        wsi.close()

        return focus_regions
