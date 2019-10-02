import logging
import os
import sys
import time

from video.pipeline import Composition2x1, Saver, SimpleController, OpenCvPipeline, Scaling
from video.opencv import HogExtractor
from video.mirror import SwitchMirrorNormal, read_front_cam_fix_parameters

sys.path.extend([
    'D:\\Programming\\PyCharm_Projects\\OpenCvPipeline',
    'D:/Programming/PyCharm_Projects/OpenCvPipeline'
])

dir_path = os.path.dirname(os.path.realpath(__file__))

my_paths = "../data/front_cam_fix_parameters.csv"
fix_parameters = read_front_cam_fix_parameters(my_paths)

orig_path = "../data/orig/{0}{1}"
restored_path = "../data/hog/{0}{1}"


def restore_single(video_id: str):
    filename, file_extension = os.path.splitext(video_id)

    input_file = orig_path.format(filename, file_extension)
    output_restored = restored_path.format(filename, file_extension)
    output_composition = restored_path.format(filename + "_composition", file_extension)

    print("Input video:", input_file)
    print("Restored video:", output_restored)
    print("Composition video:", output_composition)

    steps = []

    step_switch_mirror = SwitchMirrorNormal(fix_parameters[video_id], 16)
    steps.append(step_switch_mirror)

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
    cv_pipeline.init(input_file, output_composition)
    # cv_pipeline.apply(start_frame=2100)
    # cv_pipeline.apply(start_frame=80000, num_frames=250)
    cv_pipeline.apply(start_frame=0)

    return


if __name__ == "__main__":
    # Initialize the logging
    logging.basicConfig(level=logging.INFO)

    # Read the arguments needed: Only filenames separated with a "," are expected
    p_ids = str(sys.argv[1])
    print(p_ids)

    for p_id in p_ids.split(","):
        t = time.time()
        restore_single(p_id)
        print("\nElapsed time (s): %0.2f" % (time.time() - t))
