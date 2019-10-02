class BaseComposition:
    """
    A composition can be used to view the output of multiple keras_tf in parallel.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the step with all relevant attributes.

        :param width:
            Width of the final composition.
        :param height:
            Height of the final composition.
        """

        self.width = width
        self.height = height

    def get_np_size(self) -> (int, int):
        """
        Get the composition's shape for an numpy-array.

        :return:
            Return the dimension with a tuple as: (height, width)
        """

        return tuple((self.height, self.width))

    def get_size(self):
        """
        Get the composition's shape for the opencv-classes.

        :return:
            Return the dimension with a tuple as: (width, height)
        """

        return tuple((self.width, self.height))

    def combine(self):
        """
        Combine all the output of the given keras_tf into a single "frame" for displaying or even further processing.

        :return:
            Frame with all specified keras_tf combined.
        """

        return ()
