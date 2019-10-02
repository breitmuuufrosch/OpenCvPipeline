from typing import Dict

import cv2
import numpy as np

from ..pipeline.base_step import EditingStep


class SimpleInPainter(EditingStep):
    def __init__(self, use_static_mask: bool, static_mask: str = None, dilate_kernel_size: tuple = (1, 1),
                 inpaint_neighbors: int = 3):
        super(SimpleInPainter, self).__init__()

        if use_static_mask and static_mask is None:
            raise ValueError("use_static_mask is set to 'True' but no static_mask given")

        self.use_static_mask = use_static_mask
        self.mask = cv2.imread(static_mask) if use_static_mask else None
        self.kernel_size = dilate_kernel_size
        self.dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
        self.inpaint_neighbors = inpaint_neighbors

    def apply(self, framenumber: int, frame):
        if not self.use_static_mask:
            idx = np.all(frame <= 10, 2)
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.mask[idx] = 255

            self.mask = cv2.dilate(self.mask, self.dilate_kernel, iterations=2)
            cv2.imshow("inpainting_mask", self.mask)

        self.result = cv2.inpaint(frame, self.mask, self.inpaint_neighbors, cv2.INPAINT_NS)

        return self.result
