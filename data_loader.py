from os import path

import cv2
import numpy as np

from utils.landmarks import procrustes_transform


class RadioGraph(object):
    def __init__(self, data_path, index):
        self.index = index
        rad_path = path.join(data_path, "Radiographs/{:0>2}.tif".format(index))
        self.image = cv2.imread(rad_path)
        self.teeth = []
        self.loadteeth(data_path)

    def loadteeth(self, data_path):
        for i in range(1, 9):
            self.teeth.append(Tooth.from_file(data_path, self.index, i))


class Tooth(object):
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.pr_landmarks = procrustes_transform(self.landmarks)

    def to_img(self):
        # TODO
        return np.ones((100, 100, 1), dtype=np.uint8)

    @staticmethod
    def read_landmarks(data_path, index_rad, index_tooth):

        tooth_path = path.join(data_path,
                               "Landmarks",
                               "original",
                               "landmarks{}-{}.txt".format(index_rad,
                                                           index_tooth))
        landmarks = [[], []]
        with open(tooth_path) as f:
            for line in f:
                landmarks[0].append((int(float(line))))
                landmarks[1].append(int(float(next(f))))
        return np.array(landmarks)

    @classmethod
    def from_file(cls, data_path, index_rad, index_tooth):
        landmarks = cls.read_landmarks(data_path, index_rad, index_tooth)
        return cls(landmarks)
