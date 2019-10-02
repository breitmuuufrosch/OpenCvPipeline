import logging
import os
import sys
import time

from video.pipeline import Composition2x1, Saver, SimpleController, OpenCvPipeline, Scaling
from video.opencv import HogExtractor

sys.path.extend([
    'D:\\Programming\\PyCharm_Projects\\OpenCvPipeline',
    'D:/Programming/PyCharm_Projects/OpenCvPipeline'
])

dir_path = os.path.dirname(os.path.realpath(__file__))

restored_path = "../data/hog/webcam_{0}{1}"


def restore_single(video_id: str):
    filename, file_extension = os.path.splitext(video_id)

    output_restored = restored_path.format(filename, file_extension)
    output_composition = restored_path.format(filename + "_composition", file_extension)

    print("Input video:", "Webcam")
    print("Restored video:", output_restored)
    print("Composition video:", output_composition)

    steps = []

    step_scaling = Scaling(0.5)
    steps.append(step_scaling)

    step_hog = HogExtractor(orientations=9, pixels_per_cell=8)
    steps.append(step_hog)

    step_save = Saver(output_restored)
    steps.append(step_save)

    composition = Composition2x1(
        640, 360,
        step_scaling.get_result,
        step_hog.get_result
    )

    controller = [SimpleController()]

    cv_pipeline = OpenCvPipeline(steps, controller, composition, True)
    cv_pipeline.init(0, output_composition)
    cv_pipeline.apply(start_frame=0)

    return


if __name__ == "__main__":
    # Initialize the logging
    logging.basicConfig(level=logging.INFO)

    # Read the arguments needed: Only filename for the output file
    p_id = str(sys.argv[1])

    t = time.time()
    restore_single(p_id)
    print("\nElapsed time (s): %0.2f" % (time.time() - t))
