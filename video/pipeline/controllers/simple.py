from ..cv_pipeline import OpenCvPipeline, OpenCvPipelineState
from ..base_controller import KeyboardHandler


class SimpleController(KeyboardHandler):
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

        return False
