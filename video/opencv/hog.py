import cv2

import numpy as np
from skimage.feature import hog

from ..pipeline.base_step import EditingStep


class HogExtractor(EditingStep):
    """
    Extract the histogram of gradients on the image.
    """

    def __init__(self, orientations: int = 9, pixels_per_cell: int = 16):
        """
        Initialize the step with all relevant attributes.
        """

        super(HogExtractor, self).__init__()

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell

        self.input_frame = None

    def apply(self, framenumber: int, frame):
        """
        Apply the hog on the image and show it as a result.
        #TODO: Possibility to save into csv

        :param framenumber:
            Current frame-number.
        :param frame:
            Current frame to be processed. Either read directly from the stream or edited by a previous step.
        :return:
            Frame after the step processed/edited it.
        """
        self.input_frame = frame

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(frame, orientations=self.orientations,
                            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                            cells_per_block=(2, 2), visualize=True, block_norm="L1")

        print(fd.shape)
        # exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        max_value = np.max(hog_image)

        if max_value > 0:
            hog_image = 255 * hog_image / max_value

        hog_image = 255 - hog_image

        self.result = cv2.cvtColor(hog_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return self.result

    def get_tiles(self):
        current_x = 0
        current_y = 0

        img = np.array(self.input_frame)

        while current_x < self.width:
            cv2.line(img, (current_x, 0), (current_x, self.height), (255, 0, 0), 1)

            current_x += self.pixels_per_cell

        while current_y < self.height:
            cv2.line(img, (0, current_y), (self.width, current_y), (255, 0, 0), 1)

            current_y += self.pixels_per_cell

        return img
