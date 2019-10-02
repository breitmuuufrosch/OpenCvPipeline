from copy import deepcopy
import numpy as np

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import gc
from keras import backend
import tensorflow as tf

sys.stderr = stderr

from .pconv_model import PConvUnet


class PConvFaceRestoration:
    def __init__(self, model_path: str = None):
        self.model_path = model_path

        self.model = PConvUnet()
        self.model.load(self.model_path, train_bn=False)

        # print(self.model.model.count_params())
        # self.model.model.summary()

    def restore(self, img, mask):
        # Create masked array
        img = np.array(img) / 255

        # As for the mask only 0 and 1 should be used, set all values larger than 0 to 1
        mask[mask > 0] = 1

        # Process sample
        chunked_images = self.model.dimension_preprocess(deepcopy(img))
        chunked_masks = self.model.dimension_preprocess(deepcopy(mask))
        pred_image = self.model.predict([chunked_images, chunked_masks])
        reconstructed_image = self.model.dimension_postprocess(pred_image, img)

        # Convert the reconstructed image back to a uint with range [0, 255]
        return np.array(reconstructed_image*255, dtype=np.uint8)

    def close(self):
        # TODO: HOW TO CLEAN UP??????
        self.model.unload()
        del self.model
        backend.clear_session()
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
        gc.collect()
