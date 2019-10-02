import csv

import cv2

import numpy as np

from video.pipeline.base_step import EditingStep


class MovementExtractor(EditingStep):
    def __init__(self, use_gray_scale: bool, difference_threshold: int, adaption: float, use_mhi: bool,
                 mhi_decay: int = 20, gaussian_size: int = 3, csv_path: str = None):
        super().__init__()

        self.difference_threshold = difference_threshold
        self.background_adaption = 1 - adaption
        self.foreground_adaption = adaption
        self.use_mhi = use_mhi
        self.mhi_decay = mhi_decay
        self.gaussian_size = gaussian_size
        self.csv_path = csv_path
        self.use_gray_scale = use_gray_scale

        self.current_energy = 0
        self.energy_total = 0
        self.background = None
        self.delta = None
        self.thresh = None
        self.mhi = None

        self.csv_file = None
        self.csv_writer = None

    def init(self, length: int, fps: float, width: int, height: int):
        if self.csv_path is not None:
            self.csv_file = open(self.csv_path, 'wt', newline='')
            self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            self.csv_writer.writerow(["frame", "fidgetting_mhi", "fidgetting_threshold"])

        return super(MovementExtractor, self).init(length, fps, width, height)

    def apply(self, framenumber, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.use_gray_scale else frame
        gray = cv2.GaussianBlur(gray, (self.gaussian_size, self.gaussian_size), 0)

        if self.background is None:
            self.background = gray

        self.delta = cv2.absdiff(self.background.astype(np.uint8), gray)
        self.thresh = cv2.threshold(self.delta, self.difference_threshold, 255, cv2.THRESH_BINARY)[1]

        if self.mhi is None:
            self.mhi = self.thresh
        else:
            self.mhi = self.decay(self.mhi)
            self.mhi[self.thresh > 0] = 255

        current_energy_m = np.mean(self.mhi)
        current_energy_t = np.mean(self.thresh)

        if self.csv_file is not None:
            self.csv_writer.writerow([framenumber, current_energy_m, current_energy_t])

        if self.use_mhi:

            self.current_energy = current_energy_m
            self.energy_total += self.current_energy

            self.result = cv2.cvtColor(self.mhi, cv2.COLOR_GRAY2BGR) if self.use_gray_scale else self.mhi
        else:
            self.current_energy = current_energy_t
            self.energy_total += self.current_energy

            self.result = cv2.cvtColor(self.thresh, cv2.COLOR_GRAY2BGR) if self.use_gray_scale else self.thresh

        self.background = self.background_adaption * self.background + self.foreground_adaption * gray

        return self.result

    def print(self, frame):
        cv2.putText(frame, "Energy: {0:5.2f}".format(self.current_energy), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

        cv2.putText(frame, "Total: {0:5.2f}".format(self.energy_total), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

        return frame

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()

    def decay(self, delta):
        delta[delta >= self.mhi_decay] -= self.mhi_decay
        delta[delta < self.mhi_decay] = 0

        return delta

    def get_background(self):
        if self.use_gray_scale:
            return cv2.cvtColor(self.background.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            return self.background.astype(np.uint8)

    def get_thresh(self):
        return cv2.cvtColor(self.thresh, cv2.COLOR_GRAY2BGR) if self.use_gray_scale else self.thresh

    def get_mhi(self):
        return cv2.cvtColor(self.mhi, cv2.COLOR_GRAY2BGR) if self.use_gray_scale else self.mhi

    def get_delta(self):
        return cv2.cvtColor(self.delta, cv2.COLOR_GRAY2BGR) if self.use_gray_scale else self.delta
