import cv2

import numpy as np

from ..base_composition import BaseComposition


class Composition2x2(BaseComposition):
    def __init__(self, single_w, single_h, tl=None, tr=None, bl=None, br=None):
        """
        Initialize the step with all relevant attributes.

        :param single_w:
            Width of a step-frame.
        :param single_h:
            Height of a step-frame.
        :param tl:
            Reference to the function which gets the frame to be displayed on top left.
        :param tr:
            Reference to the function which gets the frame to be displayed on top right.
        :param bl:
            Reference to the function which gets the frame to be displayed on bottom left.
        :param br:
            Reference to the function which gets the frame to be displayed on bottom right.
        """

        super(Composition2x2, self).__init__(2 * single_w, 2 * single_h)

        self.empty = np.zeros((single_h, single_w, 3), dtype=np.uint8)
        self.single_w = single_w
        self.single_h = single_h

        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br

    def combine(self):
        """
        Combine all the output of the given keras_tf into a single "frame" for displaying or even further processing.

        :return:
            Frame with all specified keras_tf combined.
        """

        top_left = cv2.resize(self.tl(), (self.single_w, self.single_h)) if self.tl is not None else self.empty
        top_right = cv2.resize(self.tr(), (self.single_w, self.single_h)) if self.tr is not None else self.empty
        bot_left = cv2.resize(self.bl(), (self.single_w, self.single_h)) if self.bl is not None else self.empty
        bot_right = cv2.resize(self.br(), (self.single_w, self.single_h)) if self.br is not None else self.empty

        row1 = np.hstack((top_left, top_right))
        row2 = np.hstack((bot_left, bot_right))

        return np.vstack((row1, row2))
