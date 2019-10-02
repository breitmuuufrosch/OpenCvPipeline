from typing import Dict

import cv2
import numpy as np

from ..pipeline.base_step import EditingStep


class MirrorInPainter(EditingStep):
    def __init__(self, fix_parameters: Dict[str, int], inpaint_neighbors: int = 3):
        super(MirrorInPainter, self).__init__()

        self.fix_parameters = fix_parameters
        self.inpaint_neighbors = inpaint_neighbors
        self.mask = None

    def init(self, length: int, fps: float, width: int, height: int) -> (int, float, int, int):
        # Extract the needed information from the fixing parameters
        mirror_radius = self.fix_parameters["mirror_radius"]
        mirror_center_y = abs(self.fix_parameters["mirror_center_y"])
        mirror_edge_padding = 10
        padding = self.fix_parameters["padding"]
        warp_bottom = self.fix_parameters["warp_bottom"]

        # To get where the missing part is, following thoughts/formula is needed:
        # - The picture of the mirror is exactly the difference between the center and the radius of the mirror
        # - As it is flipped down to fix the image, it covers the part frame_height - mirror_image_height
        # - The edge of the mirror is rather fuzzy, that's why 10 pixels of additional padding has been applied in the
        #   switch-mirror-normal-process
        # - To get the centered y-coordinate of the missing part, from the y-coordinate from previous step, half of the
        #   padding (which has been used in the switch-mirror-normal-process) has to be subtracted (closer up)
        missing_height = height - (mirror_radius - mirror_center_y) + mirror_edge_padding - (padding // 2)
        missing_size = padding  # + 6

        # Calculate the points which are used to draw the line which covers the missing part
        missing_left = (-missing_size, missing_height)
        missing_right = (width + missing_size, missing_height)
        missing_points = np.reshape(np.array([missing_left, missing_right]), (2, 1, 2))

        left_poly = np.array([
            [0, 0],
            [0, missing_height - (padding // 2)],
            [warp_bottom, missing_height - (padding // 2)]
        ], dtype=np.int32)
        right_poly = np.array([
            [width, 0],
            [width - warp_bottom, missing_height - (padding // 2)],
            [width, missing_height - (padding // 2)]
        ], dtype=np.int32)

        self.mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.mask, [left_poly], 255)
        cv2.fillPoly(self.mask, [right_poly], 255)
        cv2.line(self.mask, tuple(missing_points[0][0]), tuple(missing_points[1][0]), 255, missing_size)

        return super(MirrorInPainter, self).init(length, fps, width, height)

    def apply(self, framenumber: int, frame):
        # mask = np.array(self.mask)
        #
        # idx = np.all(frame == 0, 2)
        # self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        # self.mask[idx] = 255
        #
        # self.mask = cv2.dilate(self.mask, self.dilate_kernel, iterations=2)
        # cv2.imshow("inpainting_mask", self.mask)

        self.result = cv2.inpaint(frame, self.mask, self.inpaint_neighbors, cv2.INPAINT_NS)

        return self.result
