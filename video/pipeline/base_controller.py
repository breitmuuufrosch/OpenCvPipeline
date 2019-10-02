class KeyboardHandler:
    """
    Base class for handling opencv-key-events.
    """

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def handle(self, key: int) -> bool:
        """
        Handle the key-input.

        :param key:
            Pressed key-event from opencv.
        :return:
            True if the event is handled and the chain can be stopped, False if not.
        """

        return False
