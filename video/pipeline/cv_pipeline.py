import logging
from enum import Enum

import cv2

from .base_composition import BaseComposition


class OpenCvPipelineState(Enum):
    NOT_INITIALIZED = 0
    INIT = 1
    PLAY = 2
    PAUSE = 3
    STOP = 4


class OpenCvPipeline:
    def __init__(self, steps: list, controller: list, composition: BaseComposition, show_composition: bool):
        self.steps = steps
        self.controller = controller
        self.composition = composition
        self.show_composition = show_composition

        self.state: OpenCvPipelineState = OpenCvPipelineState.NOT_INITIALIZED

        self.cap, self.out = None, None
        self.length, self.fps, self.width, self.height = None, None, None, None

        self.logger = logging.getLogger("OpenCV pipeline")
        self.logger.setLevel(logging.DEBUG)

    def init(self, input_file: str, output_file: str):
        self.cap = cv2.VideoCapture(input_file)

        # Read all important information about the video which are relevant for the processing keras_tf
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info("Frames File: %i" % self.length)
        self.logger.info("FPS File: %0.3f" % self.fps)
        self.logger.info("Video-Size: %ix%i" % (self.width, self.height))

        if output_file is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, self.composition.get_size())
        else:
            self.out = None

        self.state = OpenCvPipelineState.INIT
        length, fps, width, height = self.length, self.fps, self.width, self.height

        for s in self.steps:
            length, fps, width, height = s.init(length, fps, width, height)

        for c in self.controller:
            c.init(self)

    def apply(self, start_frame: int = 0, num_frames: int = 9999999):
        if self.state is False:
            raise ValueError("Please initialize the pipeline with \"init\" before applying the keras_tf")

        if start_frame is None:
            start_frame = 0
        if num_frames is None:
            num_frames = 9999999

        start_frame = max(start_frame, 0)
        self.cap.set(1, start_frame)
        self.state = OpenCvPipelineState.PLAY

        while True:
            if self.state == OpenCvPipelineState.STOP:
                break

            if self.state == OpenCvPipelineState.PLAY:
                frame_exists, frame = self.cap.read()

                if frame_exists:
                    current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print("Frame: %5i / %5.2f%%" % (current_frame, 100 * current_frame / self.length), end="\r")

                    if current_frame > start_frame + num_frames:
                        break

                    for s in self.steps:
                        frame = s.apply(current_frame, frame)

                    final = self.composition.combine()

                    if self.show_composition:
                        cv2.imshow("final", final)

                    if self.out is not None:
                        self.out.write(final)
                else:
                    break

            key = cv2.waitKey(1)

            for c in self.controller:
                if c.handle(key):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        for s in self.steps:
            s.close()

        if self.out is not None:
            self.out.release()

    def close(self):
        for s in self.steps:
            s.close()

    def jump_to_relative_frame(self, relative_frame: int):
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = current_frame + relative_frame
        self.cap.set(1, new_frame)

    def get_state(self):
        return self.state

    def set_state(self, state: OpenCvPipelineState):
        self.state = state
