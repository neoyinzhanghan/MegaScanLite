import openslide
from PIL import Image


def crop_image_at_mpp(level_0_coord, height, width, mpp, wsi):
    """
    Crop the image at the specified mpp based on the level 0 coordinates, height, and width.
    The height and width are in pixels and the level is specified by the mpp.

    Args:
        level_0_coord (tuple): The (x, y) coordinates at level 0.
        height (int): The height of the crop in pixels at the specified mpp.
        width (int): The width of the crop in pixels at the specified mpp.
        mpp (float): The desired microns per pixel.
        wsi (openslide.OpenSlide): An opened whole slide image using OpenSlide.

    Returns:
        PIL.Image: The cropped image at the specified mpp.
    """

    # Calculate the scale factor from the desired mpp to the base level mpp
    base_mpp_x = float(wsi.properties["openslide.mpp-x"])

    base_mpp_y = float(wsi.properties["openslide.mpp-y"])

    if type(base_mpp_x) != float:
        print("Erroneous base_mpp_x", base_mpp_x)

        import sys

        sys.exit()
    scale_x = base_mpp_x / mpp
    scale_y = base_mpp_y / mpp

    # Find the best level for the desired mpp by comparing the downsampling factors
    best_level = wsi.get_best_level_for_downsample(max(scale_x, scale_y))

    # Calculate the effective downsampling at the chosen level
    downsample = wsi.level_downsamples[best_level]
    target_x = int(level_0_coord[0] / downsample)
    target_y = int(level_0_coord[1] / downsample)
    target_width = int(width / scale_x)
    target_height = int(height / scale_y)

    # Crop the image at the computed level and coordinates
    region = wsi.read_region(
        (target_x, target_y), best_level, (target_width, target_height)
    )

    # Resize the image back to the desired dimensions to correct any scaling discrepancies
    if region.size != (width, height):
        region = region.resize((width, height), resample=Image.LANCZOS)

    return region


def create_list_of_batches_from_list(lst, batch_size):
    """Create a list of batches from a list of items."""

    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]
