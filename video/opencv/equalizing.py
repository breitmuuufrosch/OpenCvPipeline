from enum import Enum

import cv2
import numpy as np

from ..pipeline.base_step import EditingStep


class EqualizerMethod(Enum):
    YUV = 1
    LAB = 1


class HistogramEqualizer(EditingStep):
    """
    Equalize the illumination.
    """

    def __init__(self, method: EqualizerMethod):
        """
        Initialize the step with all relevant attributes.
        """

        super().__init__()

        self.method = method

    def init(self, length: int, fps: float, width: int, height: int):
        """
        Initialize the step with the input parameters. Then modify the width and height with the given scaling factor
        and pass this new dimension to the next keras_tf.

        :param length:
            Number of frames of the input-video.
        :param fps:
            The fps of the input-video. Can be changed by any step in the pipeline.
        :param width:
            Width of the frame for this step.
        :param height:
            Height of the frame for this step.
        :return:
            Return the input-parameters as they are edited by the current step => step doesn't affect those properties
            in the pipeline.
        """

        return super(HistogramEqualizer, self).init(length, fps, width, height)

    def apply(self, framenumber: int, frame):
        """
        Apply the pipeline for one single frame.

        :param framenumber:
            Current frame-number.
        :param frame:
            Current frame to be processed. Either read directly from the stream or edited by a previous step.
        :return:
            Frame after the step processed/edited it.
        """

        if self.method == EqualizerMethod.YUV:
            # Equalize the histogram of the Y channel of the YUV-color-space
            # https://en.wikipedia.org/wiki/YUV
            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            self.result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        elif self.method == EqualizerMethod.LAB:
            # https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            self.result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # output_r = clahe.apply(r)
        # output_g = clahe.apply(g)
        # output_b = clahe.apply(b)

        cv2.imshow("ehm", self.result)
        cv2.waitKey(0)

        return self.result


class GammaAdjustment(EditingStep):
    """
    Equalize the illumination.
    """

    def __init__(self, gamma: float = 1.0):
        """
        Initialize the step with all relevant attributes.
        """

        super().__init__()

        self.gamma = gamma

    def init(self, length: int, fps: float, width: int, height: int):
        """
        Initialize the step with the input parameters. Then modify the width and height with the given scaling factor
        and pass this new dimension to the next keras_tf.

        :param length:
            Number of frames of the input-video.
        :param fps:
            The fps of the input-video. Can be changed by any step in the pipeline.
        :param width:
            Width of the frame for this step.
        :param height:
            Height of the frame for this step.
        :return:
            Return the input-parameters as they are edited by the current step => step doesn't affect those properties
            in the pipeline.
        """

        return super(GammaAdjustment, self).init(length, fps, width, height)

    def apply(self, framenumber: int, frame):
        """
        Apply the pipeline for one single frame.

        :param framenumber:
            Current frame-number.
        :param frame:
            Current frame to be processed. Either read directly from the stream or edited by a previous step.
        :return:
            Frame after the step processed/edited it.
        """

        inverse_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in np.arange(0, 256)])

        self.result = cv2.LUT(frame.astype(np.uint8), table.astype(np.uint8))

        # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # output_r = clahe.apply(r)
        # output_g = clahe.apply(g)
        # output_b = clahe.apply(b)

        cv2.imshow("ehm", self.result)
        cv2.waitKey(0)

        return self.result
