from typing import Any


class FocusRegion:
    """=== Class Attribute ===
    - level_0_coord
    - level_0_height
    - level_0_width
    - level_0_mpp
    - mpp_to_image
    - mpp_to_PL2_score
    - wsi_path
    
    """

    def __init__(self, level_0_coord, level_0_height, level_0_width, level_0_mpp, wsi_path):
        """Initialize the focus region object."""

        self.level_0_coord = level_0_coord
        self.level_0_height = level_0_height
        self.level_0_width = level_0_width  
        self.level_0_mpp = level_0_mpp
        self.mpp_to_image = {}
        self.mpp_to_PL2_score = {}
        self.wsi_path = wsi_path

    def get_image(self, mpp, image):
        self.mpp_to_image[mpp] = image

    def get_pl2_score(self, mpp, pl2_score):
        self.mpp_to_PL2_score[mpp] = pl2_score
