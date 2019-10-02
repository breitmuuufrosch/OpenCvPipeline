import logging
import time

import sys
import os

from video.pipeline import Composition2x1, Composition2x2, Saver, SimpleController, OpenCvPipeline
from video.opencv import SimpleInPainter
from video.mirror import SwitchMirrorNormal, read_front_cam_fix_parameters

sys.path.extend([
    'D:\\Programming\\PyCharm_Projects\\OpenCvPipeline',
    'D:/Programming/PyCharm_Projects/OpenCvPipeline'
])

dir_path = os.path.dirname(os.path.realpath(__file__))

my_paths = "../data/front_cam_fix_parameters.csv"
fix_parameters = read_front_cam_fix_parameters(my_paths)

orig_path = "../data/orig/{0}{1}"
restored_path = "../data/restored_keras/{0}{1}"
inpainted_path = "../data/restored_opencv/{0}{1}"

face_predictor = "../models/opencv/shape_predictor_68_face_landmarks.dat"
restoration_model = "../models/pconv_keras/70_celeba_hq.h5"


def restore_single(video_id: str, use_simple_inpainter: bool = False):
    filename, file_extension = os.path.splitext(video_id)

    input_file = orig_path.format(filename, file_extension)
    output_restored = (inpainted_path if use_simple_inpainter else restored_path).format(filename, file_extension)
    output_composition = (inpainted_path if use_simple_inpainter else restored_path).format(filename + "_composition",
                                                                                            file_extension)

    print("Input video:", input_file)
    print("Restored video:", output_restored)
    print("Composition video:", output_composition)

    steps = []

    step_switch_mirror = SwitchMirrorNormal(fix_parameters[video_id], 16)
    steps.append(step_switch_mirror)

    # Restore the face (inpainting) based on the desired level of accuracy
    if use_simple_inpainter:
        step_inpainter = SimpleInPainter(
            use_static_mask=False, static_mask=None,
            dilate_kernel_size=(2, 2)
        )
        steps.append(step_inpainter)

        composition = Composition2x1(
            640, 360,
            step_switch_mirror.get_result,
            steps[-1].get_result
        )
    else:
        # The import is done here to only load the TensorFlow-data when it is needed.
        from video.inpainting_tf import FaceInpaintingPConvKeras

        step_face_restoration = FaceInpaintingPConvKeras(
            model_path=restoration_model,
            dat_face_predictor=face_predictor,
            fix_parameters=fix_parameters[video_id],
            always_inpaint=False,
            use_naive_inpaint_for_two_step=False
        )
        steps.append(step_face_restoration)

        composition = Composition2x2(
            640, 360,
            step_face_restoration.get_input,
            step_face_restoration.get_result_printed,
            step_face_restoration.get_aligned,
            step_face_restoration.get_result
        )

    step_save_restoration = Saver(output_restored)
    steps.append(step_save_restoration)
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
        restore_single(p_id, False)
        print("\nElapsed time (s): %0.2f" % (time.time() - t))
