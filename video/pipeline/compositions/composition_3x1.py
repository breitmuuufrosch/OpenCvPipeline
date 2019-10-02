import cv2

import numpy as np

from ..base_composition import BaseComposition


class Composition3x1(BaseComposition):
    """
    Display two keras_tf next to each other.
    """

    def __init__(self, width, height, left, middle, right, is_horizontal: bool = True):
        """
        Initialize the step with all relevant attributes.

        :param width:
            Width of the final composition.
        :param height:
            Height of the final composition.
        :param left:
            Reference to the function which gets the frame to be displayed on left.
        :param right:
            Reference to the function which gets the frame to be displayed on right.
        """

        if is_horizontal:
            super(Composition3x1, self).__init__(3 * width, height)
        else:
            super(Composition3x1, self).__init__(width, 3 * height)

        self.single_w = width
        self.single_h = height
        self.is_horizontal = is_horizontal

        self.empty = np.zeros((height, width, 3), dtype=np.uint8)
        self.left = left
        self.middle = middle
        self.right = right

    def combine(self):
        """
        Combine all the output of the given keras_tf into a single "frame" for displaying or even further processing.

        :return:
            Frame with all specified keras_tf combined.
        """

        left = cv2.resize(self.left(), (self.single_w, self.single_h)) if self.left is not None else self.empty
        middle = cv2.resize(self.middle(), (self.single_w, self.single_h)) if self.middle is not None else self.empty
        right = cv2.resize(self.right(), (self.single_w, self.single_h)) if self.right is not None else self.empty

        if self.is_horizontal:
            return np.hstack((np.hstack((left, middle)), right))
        else:
            return np.vstack((np.vstack((left, middle)), right))
