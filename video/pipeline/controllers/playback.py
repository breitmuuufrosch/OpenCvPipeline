from ..cv_pipeline import OpenCvPipeline, OpenCvPipelineState
from ..base_controller import KeyboardHandler


class PlaybackController(KeyboardHandler):
    def __init__(self):
        self.pipeline = None

    def init(self, pipeline: OpenCvPipeline):
        self.pipeline = pipeline

    def handle(self, key):
        if self.pipeline is None:
            return False

        if key & 0xFF == ord('q') or key == 27:
            self.pipeline.set_state(OpenCvPipelineState.STOP)
            return True
        if key & 0xFF == ord(','):
            self.pipeline.jump_to_relative_frame(-10)
            return True
        if key & 0xFF == ord('.'):
            self.pipeline.jump_to_relative_frame(10)
            return True
        if key & 0xFF == ord('k'):
            self.pipeline.jump_to_relative_frame(-1000)
            return True
        if key & 0xFF == ord('l'):
            self.pipeline.jump_to_relative_frame(1000)
            return True
        if key & 0xFF == ord('p'):
            if self.pipeline.get_state() == OpenCvPipelineState.PAUSE:
                new_state = OpenCvPipelineState.PLAY
            else:
                new_state = OpenCvPipelineState.PAUSE

            self.pipeline.set_state(new_state)
            return True

        return False
