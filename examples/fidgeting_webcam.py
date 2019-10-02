import logging
import os
import sys
import time

from video.pipeline import Composition2x2, Saver, SimpleController, OpenCvPipeline, Scaling
from video.fidgeting import MovementExtractor

sys.path.extend([
    'D:\\Programming\\PyCharm_Projects\\OpenCvPipeline',
    'D:/Programming/PyCharm_Projects/OpenCvPipeline'
])

dir_path = os.path.dirname(os.path.realpath(__file__))

restored_path = "../data/fidgeting/webcam_{0}{1}"
fidgeting_csv = "../data/fidgeting/webcam_{0}.csv"


def restore_single(video_id: str):
    filename, file_extension = os.path.splitext(video_id)

    output_restored = restored_path.format(filename, file_extension)
    output_composition = restored_path.format(filename + "_composition", file_extension)
    fidgeting_path = fidgeting_csv.format(filename)

    print("Input video:", "Webcam")
    print("Restored video:", output_restored)
    print("Composition video:", output_composition)

    steps = []

    step_scaling = Scaling(0.5)
    steps.append(step_scaling)

    step_fidgeting = MovementExtractor(
        use_gray_scale=True,
        difference_threshold=30,
        adaption=0.1,
        use_mhi=False,
        mhi_decay=20,
        gaussian_size=3,
        csv_path=fidgeting_path
    )
    steps.append(step_fidgeting)

    step_save = Saver(output_restored)
    steps.append(step_save)

    composition = Composition2x2(
        640, 360,
        step_scaling.get_result,
        step_fidgeting.get_background,
        step_fidgeting.get_mhi,
        step_fidgeting.get_result_printed
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
