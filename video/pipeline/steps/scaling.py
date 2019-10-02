import cv2

from ..base_step import EditingStep


class Scaling(EditingStep):
    """
    Scale the frame. (To speed up computation for other keras_tf => faster testing)
    """

    def __init__(self, scale: float):
        """
        Initialize the step with all relevant attributes.

        :param scale:
            Scaling factor.
        """

        super().__init__()

        self.scale = scale

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
            Return the input-parameters as they are edited by the current step => return adjusted size
        """

        super(Scaling, self).init(length, fps, width, height)

        return length, fps, int(width * self.scale), int(height * self.scale)

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

        self.result = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        return self.result
