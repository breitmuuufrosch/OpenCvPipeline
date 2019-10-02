from __future__ import annotations

from typing import List


class Roi:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        """
        Initialize the ROI.

        :param x1:
            Leftmost coordinate.
        :param y1:
            Topmost coordinate.
        :param x2:
            Rightmost coordinate.
        :param y2:
            Bottommost coordinate.
        """

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def top_left(self):
        """
        Get the top-left coordinate.

        :return:
            Top-left coordinate as a tuple(x/y).
        """

        return tuple((self.x1, self.y1))

    def bottom_right(self):
        """
        Get the bottom-right coordinate.

        :return:
            Bottom-right coordinate as a tuple(x/y).
        """

        return tuple((self.x2, self.y2))

    def top(self):
        """
        Get the topmost coordinate.

        :return:
            Topmost coordinate.
        """

        return self.y1

    def left(self):
        """
        Get the leftmost coordinate.

        :return:
            Leftmost coordinate.
        """

        return self.x1

    def bottom(self):
        """
        Get the bottommost coordinate.

        :return:
            Bottommost coordinate.
        """

        return self.y2

    def right(self):
        """
        Get the rightmost coordinate.

        :return:
            Rightmost coordinate.
        """

        return self.x2

    def width(self):
        """
        Get the width.

        :return:
            Width of the ROI.
        """

        return self.x2 - self.x1

    def height(self):
        """
        Get the height.

        :return:
            Height of the ROI.
        """

        return self.y2 - self.y1

    def corners(self, padding: int) -> List[tuple]:
        """
        Returns the corners in the order: top-left, top-right, bottom-right, bottom-left.

        :param padding:
            Amount to expand/shrink the ROI.
        :return:
            List of corners in the order: top-left, top-right, bottom-right, bottom-left.
        """

        x1 = self.x1 - padding
        y1 = self.y1 - padding
        x2 = self.x2 + padding
        y2 = self.y2 + padding

        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def add_padding(self, padding: int) -> Roi:
        """
        Add a padding around the ROI to expand it with a given value. If the value is negative, the ROI is shrunk.

        :param padding:
            Amount to expand/shrink the ROI.
        :return:
            The validated ROI.
        """

        self.x1 -= padding
        self.y1 -= padding
        self.x2 += padding
        self.y2 += padding

        return self

    def validate_roi_move(self, width: int, height: int) -> Roi:
        """
        Validate if the ROI is completely in the possible window and adjust it if not. It's adjusted by moving the ROI
        so that it fits completely inside the frame.
        It can happen that the ROI is not aligned correctly anymore with the desired features. But the size doesn't
        change.

        :param width:
            Width of the video-frame.
        :param height:
            Height of the video-frame
        :return:
            The validated ROI.
        """

        if self.width() > width:
            raise ValueError("ROI too large. Width of ROI: {0} / Width of image: {1}".format(self.width(), width))

        if self.height() > height:
            raise ValueError("ROI too large. Height of ROI: {0} / Height of image: {1}".format(self.height(), height))

        if self.x1 < 0:
            self.x2 -= self.x1
            self.x1 = 0
        elif self.x2 >= width:
            self.x1 -= (self.x2 - width)
            self.x2 = width - 1

        if self.y1 < 0:
            self.y2 -= self.y1
            self.y1 = 0
        elif self.y2 >= height:
            self.y1 -= (self.y2 - height)
            self.y2 = height - 1

        return self

    def validate_roi_cut(self, width: int, height: int) -> Roi:
        """
        Validate if the ROI is completely in the possible window and adjust it if not. It's adjusting by cutting the
        ROI so that its coordinates are completely inside the frame.
        It can happen that the ROI is not a square anymore.

        :param width:
            Width of the video-frame.
        :param height:
            Height of the video-frame
        :return:
            The validated ROI.
        """

        self.x1 = max(self.x1, 0)
        self.x2 = min(self.x2, width)
        self.y1 = max(self.y1, 0)
        self.y2 = min(self.y2, height)

        return self

    def make_it_square(self) -> Roi:
        """
        In case if the ROI is not yet a square (height is equal width), adjust it to be square.

        :return:
            The validated ROI.
        """
        w = self.width()
        h = self.height()

        if w == h:
            return self

        c = (self.left() + w // 2, self.top() + h // 2)
        new_size = max(w, h) // 2

        self.x1 = c[0] - new_size
        self.x2 = c[0] + new_size
        self.y1 = c[1] - new_size
        self.y2 = c[1] + new_size

        return self

    def apply(self, frame, padding: int = 0):
        """
        Apply the ROI on the frame by returning the ROI of the frame.

        :param frame:
            Video frame to apply the ROI.
        :param padding:
            Amount to expand/shrink the ROI.
        :return:
            ROI of the frame.
        """
        return frame[self.y1-padding:self.y2+padding, self.x1-padding:self.x2+padding]
