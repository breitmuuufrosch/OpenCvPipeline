from enum import IntEnum

import dlib

import cv2
import imutils
from imutils import face_utils
from typing import Dict, Tuple, Union, Any

import numpy as np
from scipy.spatial import distance

from .keras_tf.pconv_predictor import PConvFaceRestoration
from ..pipeline.base_step import EditingStep
from ..pipeline.roi import Roi


class FaceFoundState(IntEnum):
    """ Specifies the current log-level to print to the console. """
    NO_FACE = 0
    FOUND_DIRECTLY = 1
    FOUND_INDIRECTLY = 2


# noinspection PyMethodMayBeStatic
class FaceInpaintingPConvKeras(EditingStep):
    def __init__(self, model_path: str, dat_face_predictor: str, fix_parameters: Dict[str, int], always_inpaint: bool,
                 use_naive_inpaint_for_two_step: bool):
        super(FaceInpaintingPConvKeras, self).__init__()

        self.model_path = model_path
        self.dat_face_predictor = dat_face_predictor
        self.fix_parameters = fix_parameters
        self.always_inpaint = always_inpaint
        self.use_naive_inpaint_for_two_step = use_naive_inpaint_for_two_step

        self.face_restorer, self.face_detector, self.face_predictor = None, None, None

        self.use_corrected_roi = True
        self.rotation_m, self.rotation_inv = np.eye(N=2, M=3), np.eye(N=2, M=3)

        self.count = {FaceFoundState.NO_FACE: 0, FaceFoundState.FOUND_DIRECTLY: 0, FaceFoundState.FOUND_INDIRECTLY: 0}

        self.missing_size = 0
        self.missing_points = []
        self.missing_points_1, self.missing_points_2 = (), ()

        self.last_roi = None
        self.last_shape = None
        self.last_right_eye_center, self.last_left_eye_center = None, None

        self.frame_input = None
        self.frame_aligned = None
        self.mask_mirror = None
        self.mask_aligned = None
        self.mask_original = None

        self.enable_image_show = False
        self.wait_key = 10 * 1000

        self.face_found = FaceFoundState.NO_FACE

        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def init(self, length: int, fps: float, width: int, height: int):
        # Extract the needed information from the fixing parameters
        mirror_radius = self.fix_parameters["mirror_radius"]
        mirror_center_y = abs(self.fix_parameters["mirror_center_y"])
        mirror_edge_padding = 10
        padding = self.fix_parameters["padding"]
        warp_bottom = self.fix_parameters["warp_bottom"]

        # To get where the missing part is, following thoughts/formula is needed:
        # - The picture of the mirror is exactly the difference between the center and the radius of the mirror
        # - As it is flipped down to fix the image, it covers the part frame_height - mirror_image_height
        # - The edge of the mirror is rather fuzzy, that's why 10 pixels of additional padding has been applied in the
        #   switch-mirror-normal-process
        # - To get the centered y-coordinate of the missing part, from the y-coordinate from previous step, half of the
        #   padding (which has been used in the switch-mirror-normal-process) has to be subtracted (closer up)
        missing_top = height - (mirror_radius - mirror_center_y) + mirror_edge_padding - padding
        missing_height = height - (mirror_radius - mirror_center_y) + mirror_edge_padding - (padding // 2)
        self.missing_size = padding + 6

        # Calculate the points which are used to draw the line which covers the missing part
        missing_left = (-self.missing_size, missing_height)
        missing_right = (width + self.missing_size, missing_height)
        self.missing_points = np.reshape(np.array([missing_left, missing_right]), (2, 1, 2))
        self.missing_points_1 = tuple(self.missing_points[0][0])
        self.missing_points_2 = tuple(self.missing_points[1][0])

        self.mask_original = self.generate_mask(width, height, 3)

        self.face_restorer = PConvFaceRestoration(model_path=self.model_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(self.dat_face_predictor)

        left_poly = np.array([[0, 0], [0, missing_top], [warp_bottom, missing_top]], dtype=np.int32)
        right_poly = np.array([[width, 0], [width - warp_bottom, missing_top], [width, missing_top]], dtype=np.int32)

        self.mask_mirror = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.mask_mirror, [left_poly], 255)
        cv2.fillPoly(self.mask_mirror, [right_poly], 255)
        cv2.line(self.mask_mirror, self.missing_points_1, self.missing_points_2, 255, self.missing_size)

        return super(FaceInpaintingPConvKeras, self).init(length, fps, width, height)

    def apply(self, framenumber: int, frame):
        self.frame_input = np.array(frame)
        frame = np.array(frame)
        detection_result = self.detect_face(frame)

        if detection_result is False and (self.last_roi is not None or self.use_naive_inpaint_for_two_step):
            frame_inpainted = np.array(frame)

            if self.use_naive_inpaint_for_two_step:
                frame_inpainted = self.inpaint_background(frame, None, None)
            else:
                # Align the frame and mask as for the last known working frame
                frame_aligned = self.apply_rotation(frame, self.rotation_m)
                mask_aligned = self.generate_rotated_mask(frame, self.rotation_m)

                # Inpaint with the last known working rotation and roi => hopefully still matches well
                self.restore_face(frame_inpainted, frame_aligned, mask_aligned, self.last_roi, self.rotation_inv)

            # Frame_inpainted = self.apply_rotation(frame_inpainted, self.rotation_m)
            self.optional_show_image("frame_inpainted", frame_inpainted, 1)

            # Redo the face detection in the hope for the rotated and inpainted frame the face can be found correctly.
            # In case it is found => re-inpaint with the more accurate newly found parameters
            detection_result = self.detect_face(frame_inpainted)

            if detection_result is not False:
                face_found, shape, rotation_m, rotation_inv, roi, eyes = detection_result

                # Combine new rotation matrices
                # rotation_indirect, rotation_indirect_inv = self.combine_rotations(self.rotation_m, self.rotation_inv)
                rotation_indirect, rotation_indirect_inv = rotation_m, rotation_inv

                # Align the frame and mask to have the eyes horizontally
                frame_aligned = self.apply_rotation(frame, rotation_indirect)
                mask_aligned = self.generate_rotated_mask(frame, rotation_indirect)
                # self.optional_show_image("new_aligned", frame_aligned, 1)
                # self.optional_show_image("new_mask", mask_aligned * 255, 1)

                # Inpaint the missing part of the face which is detected in the specified ROI
                self.restore_face(frame, frame_aligned, mask_aligned, roi, rotation_indirect_inv)
                frame = self.inpaint_background(frame, roi, rotation_inv)
                # self.optional_show_image("new_restored", frame, 0)

                self.face_found = FaceFoundState.FOUND_INDIRECTLY
                self.save_detection_result(detection_result, frame_aligned=frame_aligned, mask_aligned=mask_aligned,
                                           new_rotation_m=rotation_indirect, new_rotation_inv=rotation_indirect_inv)
            else:
                self.face_found = FaceFoundState.NO_FACE
                self.frame_aligned = frame

                if self.always_inpaint:
                    frame = frame_inpainted
                    frame = self.inpaint_background(frame, self.last_roi, self.rotation_inv)

        elif detection_result is False and self.last_roi is None:
            self.face_found = FaceFoundState.NO_FACE
            self.frame_aligned = frame

            if self.always_inpaint:
                frame = self.inpaint_background(frame)

        else:
            face_found, shape, rotation_m, rotation_inv, roi, eyes = detection_result

            # Align the frame and mask to have the eyes horizontally
            frame_aligned = self.apply_rotation(frame, rotation_m)
            mask_aligned = self.generate_rotated_mask(frame, rotation_m)

            # Inpaint the missing part of the face which is detected in the specified ROI
            self.restore_face(frame, frame_aligned, mask_aligned, roi, rotation_inv)
            frame = self.inpaint_background(frame, roi, rotation_inv)

            self.face_found = FaceFoundState.FOUND_DIRECTLY
            self.save_detection_result(detection_result, frame_aligned=frame_aligned, mask_aligned=mask_aligned)

        self.count[self.face_found] += 1
        self.result = frame

        return self.result

    def print(self, frame):
        if self.last_roi is not None:
            cv2.rectangle(frame, self.last_roi.top_left(), self.last_roi.bottom_right(), (0, 255, 0), 1)

        if self.last_shape is not None:
            # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
            for (i, (x, y)) in enumerate(self.last_shape):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if self.last_left_eye_center is not None:
            cv2.circle(frame, tuple(self.last_left_eye_center.astype(int)), 10, (255, 0, 0), -1)
            cv2.circle(frame, tuple(self.last_right_eye_center.astype(int)), 10, (255, 0, 0), -1)

        if self.face_found == FaceFoundState.NO_FACE:
            ff_string = "No face found"
        elif self.face_found == FaceFoundState.FOUND_DIRECTLY:
            ff_string = "Face found: directly"
        elif self.face_found == FaceFoundState.FOUND_INDIRECTLY:
            ff_string = "Face found: indirectly"
        else:
            ff_string = "WHAT JUST HAPPENED"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1200, 70), (33, 33, 33), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        cv2.putText(frame, ff_string, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "No face: {0}, Directly: {1}, Indirectly: {2}".format(*self.count.values()), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return frame

    def close(self):
        if self.face_restorer is not None:
            self.face_restorer.close()

    def save_detection_result(self, detection_result, frame_aligned, mask_aligned, new_rotation_m=None,
                              new_rotation_inv=None):
        _, shape, rotation_m, rotation_inv, roi, eyes = detection_result

        self.frame_aligned = frame_aligned
        self.mask_aligned = mask_aligned
        self.last_shape = shape
        self.last_roi = roi
        self.rotation_m = new_rotation_m if new_rotation_m is not None else rotation_m
        self.rotation_inv = new_rotation_inv if new_rotation_inv is not None else rotation_inv
        self.last_left_eye_center, self.last_right_eye_center = eyes

    def restore_face(self, frame, frame_aligned, mask_aligned, roi, rotation_inv):
        face = roi.apply(frame_aligned)
        mask = roi.apply(mask_aligned)

        idx_mask = np.where(mask == 0)
        # face[idx_mask] = 0

        do_scale = False if roi.width() == 512 and roi.height() == 512 else True
        size_orig = (roi.width(), roi.height())

        if do_scale:
            nn_face = cv2.resize(face, (512, 512))
            nn_mask = cv2.resize(mask, (512, 512))
        else:
            nn_face = face
            nn_mask = mask

        restored_face = self.face_restorer.restore(nn_face, nn_mask)

        if do_scale:
            restored_face = cv2.resize(restored_face, size_orig)

        face[idx_mask] = restored_face[idx_mask]

        frame_inpainted = self.apply_rotation(frame_aligned, rotation_inv)

        idx_mask = np.where(self.mask_original == 0)
        frame[idx_mask] = frame_inpainted[idx_mask]
        self.optional_show_image("restored_face", restored_face)
        self.optional_show_image("restored_face2", face)
        self.optional_show_image("restored_inpainted", frame_inpainted)
        self.optional_show_image("restored_frame", frame, 10000)

    def inpaint_background(self, frame, roi: Roi = None, rotation_inv=None) -> np.ndarray:
        mask = np.array(self.mask_mirror)
        padding = -10

        if roi is not None:
            roi_points = np.reshape(np.array(roi.corners(padding)), (4, 1, 2))
            t_mask = cv2.transform(roi_points, rotation_inv)
            roi_points = np.array([[t_mask[0][0], t_mask[1][0], t_mask[2][0], t_mask[3][0]]], dtype=np.int32)

            cv2.fillPoly(mask, roi_points, 0)

        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

    def detect_face(self, frame: np.ndarray) -> Union[Tuple[bool, np.ndarray, None, None, Roi, Tuple[Any, Any]], bool]:
        # detect faces in the gray-scale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detector(gray, 0)

        # As only one face is expected, only look at the first if there is any found
        if len(rects) > 0:
            rect = rects[0]

            # determine the facial landmarks for the face region and convert the facial landmark (x, y)-coordinates
            shape = self.face_predictor(frame, rect)
            shape = imutils.face_utils.shape_to_np(shape)

            left_eye = shape[self.l_start:self.l_end]
            right_eye = shape[self.r_start:self.r_end]

            # get the angle between the eyes
            left_eye_center = left_eye.mean(axis=0)
            right_eye_center = right_eye.mean(axis=0)
            eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
            eye_distance = distance.cdist(np.array([left_eye_center]), np.array([right_eye_center]), "euclidean")[0][0]

            # store the eye-centers
            self.last_left_eye_center = left_eye_center
            self.last_right_eye_center = right_eye_center

            d_y = np.mean([left_eye[0][1] - left_eye[3][1], right_eye[0][1] - right_eye[3][1]])
            d_x = np.mean([left_eye[0][0] - left_eye[3][0], right_eye[0][0] - right_eye[3][0]])

            angle = np.degrees(np.arctan2(d_y, d_x)) - 180

            # Create the rotation matrix for aligning the face as well the inverse to rotate the frame in the orig
            # position
            rotation_m = cv2.getRotationMatrix2D(tuple(eye_center.astype(int)), angle, 1)
            rotation_inv = cv2.getRotationMatrix2D(tuple(eye_center.astype(int)), -angle, 1)

            # Calculate the ROI based on the eye-center (which remains after rotation) and the eye-distance.
            roi = self.calculate_roi(eye_distance, eye_center)

            return True, shape, rotation_m, rotation_inv, roi, (left_eye_center, right_eye_center)

        else:
            return False

    def apply_rotation(self, frame: np.ndarray, rotation) -> np.ndarray:
        # Rotate the frame so the eyes are aligned horizontally.
        return cv2.warpAffine(frame, rotation, None, flags=cv2.INTER_CUBIC)

    def generate_rotated_mask(self, frame: np.ndarray, rotation) -> np.ndarray:
        # Create an empty mask and then draw a line where the missing part is. To know the ends of the line,
        # the line-points from the original frame are rotated in the same manner as the frame.
        t_mask = cv2.transform(self.missing_points, rotation)

        mask_aligned = np.ones(frame.shape, dtype=np.uint8)
        cv2.line(mask_aligned, tuple(t_mask[0][0]), tuple(t_mask[1][0]), (0, 0, 0), self.missing_size)

        return mask_aligned

    def combine_rotations(self, rotation_m, rotation_inv):
        # Combine new rotation matrices
        rotation_first = np.array(np.vstack((self.rotation_m, [0, 0, 1])))
        rotation_second = np.array(np.vstack((rotation_m, [0, 0, 1])))
        rotation_first_inv = np.array(np.vstack((self.rotation_inv, [0, 0, 1])))
        rotation_second_inv = np.array(np.vstack((rotation_inv, [0, 0, 1])))

        rotation_indirect = (rotation_first @ rotation_second)[0:2, :]
        rotation_indirect_inv = (rotation_second_inv @ rotation_first_inv)[0:2, :]
        # rotation_indirect = (rotation_second @ rotation_first)[0:2, :]
        # rotation_indirect_inv = (rotation_first_inv @ rotation_second_inv)[0:2, :]

        return rotation_indirect, rotation_indirect_inv

    def generate_mask(self, width: int, height: int, channels: int):
        mask = np.ones((height, width, channels), dtype=np.uint8)
        cv2.line(mask, self.missing_points_1, self.missing_points_2, (0, 0, 0), self.missing_size)

        return mask

    def calculate_roi(self, eye_distance: float, eye_center: np.ndarray) -> Roi:
        """
        Calculate the correct roi
        - On the trained images, on a 512x512 picture, the eyes have a distance of 130 pixels between
        - Approximately: Left eye: 190/256, Right eye: 320/256, Mouth: 210/380 to 300/380

        :param eye_distance:
            Current eye-distance on the frame.
        :param eye_center:
            2D location on the frame where the center between both eyes is
        :return:
            Correct ROI which conforms the standards of the trained model for inpainting.
        """

        # Get the correct scale between the current and the desired distance of the eye
        scale = eye_distance / 130
        roi_x1 = int(eye_center[0] - 256 * scale)
        roi_x2 = int(eye_center[0] + 256 * scale)
        roi_y1 = int(eye_center[1] - 0.48 * (roi_x2 - roi_x1))
        roi_y2 = int(eye_center[1] + 0.52 * (roi_x2 - roi_x1))

        return Roi(roi_x1, roi_y1, roi_x2, roi_y2).validate_roi_cut(self.width, self.height)

    def get_input(self):
        return self.frame_input

    def get_aligned(self):
        return self.frame_aligned

    def get_mask_aligned(self):
        return self.mask_aligned

    def optional_show_image(self, name: str, frame: np.ndarray, wait: int = -1):
        if self.enable_image_show:
            cv2.imshow(name, frame)

            if wait >= 0:
                cv2.waitKey(wait)

    def optional_wait_key(self):
        if self.wait_key >= 0:
            cv2.waitKey(self.wait_key)
