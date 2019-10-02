import numpy as np


class EditingStep:
    """
    This is the base class for a pipeline step in the opencv-pipeline.
    """

    def __init__(self):
        """
        Initialize the step with all relevant attributes.
        """

        self.result = None

        self.length = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0

    def init(self, length: int, fps: float, width: int, height: int) -> (int, float, int, int):
        """
        Initialize the step with important information for the further processing. Each step is initialized after the
        keras_tf as they are applied in the apply-function. Therefore, if any step is scaling the image, the later keras_tf
        in the pipeline know about this and can work with the correct numbers.

        :param length:
            Number of frames of the input-video.
        :param fps:
            The fps of the input-video. Can be changed by any step in the pipeline.
        :param width:
            Width of the frame for this step.
        :param height:
            Height of the frame for this step.
        :return:
            Return the input-parameters as they are edited by the current step.
            (e.g.: if the step is scaling the image by half, it returns the new width <- width // 2)
        """

        self.length, self.fps, self.width, self.height = length, fps, width, height

        return length, fps, width, height

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

        return frame

    def print(self, frame):
        """
        Print additional information of the frame on the frame

        :param frame:
            Current frame to be printed on. Either read directly from the stream or edited by a previous step.
        :return:
            Frame with the printed information on it.
        """

        return frame

    def close(self) -> None:
        """
        Stop the processing and close all opened handlers etc if needed.
        """

        return

    def get_result(self):
        """
        Get the final frame of the current step which is passed to the next step in the pipeline.

        :return:
            Frame after the step processed/edited it.
        """

        return self.result

    def get_result_printed(self):
        """
        Get the final frame of the current step with some additional information printed on. Good for the composition-
        view to directly see some special information about the step.

        :return:
            Frame with the printed information on it.
        """

        # Make a copy of the frame, so that next keras_tf are not affected
        return self.print(np.array(self.result))
