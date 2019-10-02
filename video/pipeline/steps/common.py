import cv2

from ..base_step import EditingStep


class Viewer(EditingStep):
    """
    View the current frame at the given position in the pipeline. Can be used to supervise the process and the
    correctness of a specific step.
    """

    def __init__(self, name: str):
        """
        Initialize the step with all relevant attributes.

        :param name:
            Name of the opencv-window.
        """

        super().__init__()

        self.name = name

    def apply(self, framenumber: int, frame):
        """
        Show the image in the named-window.

        :param framenumber:
            Current frame-number.
        :param frame:
            Current frame to be processed. Either read directly from the stream or edited by a previous step.
        :return:
            Frame after the step processed/edited it.
        """

        self.result = frame
        cv2.imshow(self.name, frame)

        return self.result


class Saver(EditingStep):
    """
    Save the current frame at the end of a given output-stream.
    """

    def __init__(self, file_name: str, fps: float = None):
        """
        Initialize the step with all relevant attributes.

        :param file_name:
            Path to the file to write the stream.
        """

        super().__init__()

        self.file_name = file_name
        self.out = None
        self.forced_fps = fps

    def init(self, length: int, fps: float, width: int, height: int) -> (int, float, int, int):
        """
        Initialize the output-stream with the correct fps and shape.

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

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = self.forced_fps if self.forced_fps is not None else fps
        self.out = cv2.VideoWriter(self.file_name, fourcc, fps, (width, height))

        return super(Saver, self).init(length, fps, width, height)

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

        self.result = frame
        self.out.write(frame)

        return self.result

    def close(self):
        """
        Close the output stream
        """

        self.out.release()

        return
