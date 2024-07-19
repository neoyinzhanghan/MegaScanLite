import os
import ray
import openslide
from tqdm import tqdm
from MS.MSFocuRegion import FocusRegion
from utils import crop_image_at_mpp, create_list_of_batches_from_list
from config import *
from MS.RegionCropManager import WSICropper
from tqdm import tqdm


class MSScanner:
    """
    === Class Attributes ===
    - wsi_path
    - scan_mpp
    - best_scan_level
    - downsample_factor_from_level_0
    - level_0_width
    - level_0_height
    - scan_level_width
    - scan_level_height
    - focus_regions_level_0_height
    - focus_regions_level_0_width
    - focus_regions_scan_level_height
    - focus_regions_scan_level_width
    - focus_regions_level_0_mpp
    - focus_regions
    - level_0_coords
    - scan_mpp_coords
    - verbose
    - hoarding
    - result_dir

    """

    def __init__(
        self,
        wsi_path,
        scan_mpp,
        focus_region_level_0_height,
        focus_region_level_0_width,
        verbose=False,
        hoarding=False,
        result_dir=None,
    ):
        """Initialize the scanner object."""

        self.wsi_path = wsi_path
        self.scan_mpp = scan_mpp
        self.focus_regions = []
        self.verbose = verbose
        self.hoarding = hoarding
        self.result_dir = result_dir
        self.focus_regions_level_0_height = focus_region_level_0_height
        self.focus_regions_level_0_width = focus_region_level_0_width

        # first open the wsi
        wsi = openslide.OpenSlide(self.wsi_path)

        # get the level 0 dimensions
        level_0_width, level_0_height = wsi.dimensions
        self.level_0_width = level_0_width
        self.level_0_height = level_0_height

        # get the level 0 mpp
        self.level_0_mpp = float(wsi.properties["openslide.mpp-x"])

        # calculate the scan level downsampling rate from level 0
        self.downsample_factor_from_level_0 = self.level_0_mpp / self.scan_mpp

        # calculate the best scan level
        self.best_scan_level = wsi.get_best_level_for_downsample(
            self.downsample_factor_from_level_0
        )

        # calculate the scan level dimensions based on the level_0 dimensions
        self.scan_level_width = int(level_0_width / self.downsample_factor_from_level_0)
        self.scan_level_height = int(
            level_0_height / self.downsample_factor_from_level_0
        )

        # if the downsampling rate is not an integer, raise an error
        assert (
            self.scan_level_width.is_integer()
        ), f"Scan level width {self.scan_level_width} is not an integer"
        assert (
            self.scan_level_height.is_integer()
        ), f"Scan level height {self.scan_level_height} is not an integer"

        # if the level_0 dimension of hte focus region is not divisible by the downsample factor, raise an error
        assert round(
            self.focus_regions_level_0_height % self.downsample_factor_from_level_0, 3
        ) == float(
            0
        ), f"Focus region height {self.focus_regions_level_0_height} is not divisible by the downsample factor {self.downsample_factor_from_level_0}, when scanning at {self.scan_mpp} mpp and the level 0 dimensions are {level_0_width} x {level_0_height} at {self.level_0_mpp} mpp"
        assert round(
            self.focus_regions_level_0_width % self.downsample_factor_from_level_0, 3
        ) == float(
            0
        ), f"Focus region width {self.focus_regions_level_0_width} is not divisible by the downsample factor {self.downsample_factor_from_level_0}, when scanning at {self.scan_mpp} mpp, and the level 0 dimensions are {level_0_width} x {level_0_height} at {self.level_0_mpp} mpp"

        # calculate the focus region dimensions at the scan level
        self.focus_regions_scan_level_height = int(
            self.focus_regions_level_0_height / self.downsample_factor_from_level_0
        )

        self.focus_regions_scan_level_width = int(
            self.focus_regions_level_0_width / self.downsample_factor_from_level_0
        )

        # use the scan level dimensions to get the list of all (TL_x, TL_y) coordinates of the focus regions at level 0 and the scan level

        num_x = int(self.scan_level_width / self.focus_regions_scan_level_width)
        num_y = int(self.scan_level_height / self.focus_regions_scan_level_height)

        scan_mpp_coords = []
        level_0_coords = []

        for i in range(num_x):
            for j in range(num_y):
                TL_x = i * self.focus_regions_scan_level_width
                TL_y = j * self.focus_regions_scan_level_height

                scan_mpp_coords.append((TL_x, TL_y))

                level_0_x = int(TL_x * self.downsample_factor_from_level_0)
                level_0_y = int(TL_y * self.downsample_factor_from_level_0)

                level_0_coords.append((level_0_x, level_0_y))

        self.level_0_coords = level_0_coords
        self.scan_mpp_coords = scan_mpp_coords

        for i in tqdm(
            range(len(self.level_0_coords)), desc="Initializing FocusRegion Objects"
        ):

            level_0_coord, scan_mpp_coord = (
                self.level_0_coords[i],
                self.scan_mpp_coords[i],
            )

            focus_region = FocusRegion(
                level_0_coord=level_0_coord,
                level_0_height=self.focus_regions_level_0_height,
                level_0_width=self.focus_regions_level_0_width,
                level_0_mpp=self.level_0_mpp,
                wsi_path=self.wsi_path,
            )

            self.focus_regions.append(focus_region)

        # close the wsi
        wsi.close()

    def crop_images_at_scan_mpp(self):
        """Crop the focus regions at the scan mpp."""

        # new focus regions list
        new_focus_regions = []

        # create a list of batches of focus regions
        batches = create_list_of_batches_from_list(
            self.focus_regions, scanning_batch_size
        )

        ray.shutdown()
        ray.init()

        # create a list of croppers (num_croppers)
        croppers = [
            WSICropper.remote(self.wsi_path, self.scan_mpp) for _ in range(num_croppers)
        ]

        # progress bar for tracking the cropping process
        pbar = tqdm(total=len(self.focus_regions), desc="Cropping focus regions")

        # crop the focus regions using the async_crop_focus_region method
        for batch in batches:
            futures = [
                cropper.async_crop_focus_region.remote(region)
                for cropper, region in zip(croppers, batch)
            ]
            results = ray.get(futures)
            new_focus_regions.extend(results)

            pbar.update(len(batch))

        pbar.close()

        # shutdown ray
        ray.shutdown()

        self.focus_regions = new_focus_regions


if __name__ == "__main__":

    ray_tmp_dir = "/media/hdd1/neo/BMA_AML_lite/ray_tmp"

    if not os.path.exists(ray_tmp_dir):
        os.makedirs(ray_tmp_dir)

    os.environ["RAY_TEMP_DIR"] = ray_tmp_dir

    wsi_path = (
        "/media/hdd1/neo/BMA_AML_lite/H23-568;S13;MSK1 - 2023-08-24 22.09.58.ndpi"
    )

    # initialize the scanner object
    scanner = MSScanner(
        wsi_path=wsi_path,
        scan_mpp=scan_mpp,
        focus_region_level_0_height=focus_region_scan_mpp_height,
        focus_region_level_0_width=focus_region_scan_mpp_width,
    )

    scanner.crop_images_at_scan_mpp()
