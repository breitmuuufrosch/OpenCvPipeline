from ..base_composition import BaseComposition


class Composition1x1(BaseComposition):
    """
    Only one step can be displayed.
    """

    def __init__(self, width, height, tl):
        """
        Initialize the step with all relevant attributes.

        :param width:
            Width of the final composition.
        :param height:
            Height of the final composition.
        :param tl:
            Reference to the function which gets the frame to be displayed on top left.
        """

        super(Composition1x1, self).__init__(width, height)

        self.tl = tl

    def combine(self):
        """
        Combine all the output of the given keras_tf into a single "frame" for displaying or even further processing.

        :return:
            Frame with all specified keras_tf combined.
        """

        return self.tl()
