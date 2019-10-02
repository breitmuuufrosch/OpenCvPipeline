import cv2
import math

import numpy as np

from ..pipeline.base_step import EditingStep


class SwitchMirrorNormal(EditingStep):
    def __init__(self, fix_parameters: dict, mirror_pieces: int):
        super().__init__()

        self.circle = list([fix_parameters["mirror_center_x"], fix_parameters["mirror_center_y"],
                            fix_parameters["mirror_radius"]])
        self.padding = fix_parameters["padding"]
        self.warp_top = fix_parameters["warp_top"]
        self.warp_bottom = fix_parameters["warp_bottom"]

        # The circle has been chosen always to lie on the edge. As the edge is a little bit blurry, cut away a little
        # bit more to be safer. Safer as the image is less blurry/distorted and can be restored later
        self.circle[1] -= 10

        # Get the minimal point of the mirror to know where to flip the image
        self.line_height = self.circle[1] + self.circle[2]

        self.mirror_pieces = mirror_pieces

        self.lower_m, self.upper_m, self.warp_info = None, None, None

    def init(self, length: int, fps: float, width: int, height: int):
        # Warp perspective information for the mirror part
        upper_perspective_orig = np.float32(
            [[0, 0], [width, 0], [0, self.line_height], [width, self.line_height]])
        upper_perspective_new = np.float32(
            [[-self.warp_top, 0], [width + self.warp_top, 0], [0, self.line_height],
             [width, self.line_height]])
        self.upper_m = cv2.getPerspectiveTransform(upper_perspective_orig, upper_perspective_new)

        # Warp perspective information for the normal part
        lower_perspective_orig = np.float32(
            [[0, 0], [width, 0], [0, self.line_height], [width, self.line_height]])
        lower_perspective_new = np.float32(
            [[0, 0], [width, 0], [self.warp_bottom, self.line_height],
             [width - self.warp_bottom, self.line_height]])
        self.lower_m = cv2.getPerspectiveTransform(lower_perspective_orig, lower_perspective_new)

        # Warp mirror to have the edge aligned horizontally
        self.warp_info = self.calc_mirror_warp(width, self.circle[2], self.circle[0], self.line_height,
                                               self.mirror_pieces)

        return super(SwitchMirrorNormal, self).init(length, fps, width, height)

    def apply(self, framenumber, frame):
        lower = frame[self.line_height + self.padding:, :, :]
        upper = frame[:self.line_height, :, :]

        # Mirror part:
        # - Correct perspective (don't do this at first step => piecewise later)
        # - Mask the mirror (don't mask it anymore as the warping is done piecewise
        # - Flip the image
        # - Warp the mirror piecewise
        upper = cv2.flip(upper, 0)
        upper = self.apply_mirror_warp(upper)

        # Normal part:
        # - Correct perspective
        lower = cv2.warpPerspective(lower, self.lower_m,
                                    (self.width, (self.height - self.line_height - self.padding)))

        self.result = np.zeros(frame.shape, dtype=np.uint8)
        self.result[:(self.height - self.line_height - self.padding), :, :] = lower
        self.result[self.height - self.line_height:, :, :] = upper

        return self.result

    # noinspection PyMethodMayBeStatic
    def get_y(self, c_r, c_x):
        return int(-c_r + math.sqrt(c_r * c_r + c_x * c_x))

    def calc_mirror_warp(self, width: int, c_r: int, c_x: int, line_height: int, pieces: int = 16):
        p_width = width // pieces

        warp_info = {}

        for i in range(pieces):
            range_start = 0 if i == 0 else (i * p_width - 1)
            range_end = (i + 1) * p_width

            x1 = 0
            x2 = p_width + 1
            x3 = x2
            x4 = x1

            y1 = self.get_y(c_r, c_x - range_start)
            y2 = self.get_y(c_r, c_x - range_end)
            y3 = line_height
            y4 = line_height

            p_orig = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            p_new = np.float32([[x1, 0], [x2, 0], [x3, y3], [x4, y4]])
            p_m = cv2.getPerspectiveTransform(p_orig, p_new)

            warp_info[i] = {
                "width": p_width if i == 0 else p_width + 1,
                "height": line_height,
                "x_start": range_start,
                "x_end": range_end,
                "p_m": p_m,
                "p_orig": p_orig,
                "p_new": p_new
            }

        return warp_info

    def apply_mirror_warp(self, mirror):
        mirror_warped = np.array(mirror)

        for v in self.warp_info.values():
            p_width = v["width"]
            p_height = v["height"]
            x_start = v["x_start"]
            x_end = v["x_end"]
            p_m = v["p_m"]

            part = np.array(mirror[:, x_start:x_end])
            mirror_warped[:, x_start:x_end] = cv2.warpPerspective(part, p_m, (p_width, p_height))

        return mirror_warped
