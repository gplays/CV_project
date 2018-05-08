import numpy as np

from data_loader import Tooth
from utils.landmarks import to_polar, to_cart


def run_pca(radiographs, polar=False):
    teeth_data = np.zeros((2, 8, 14, 40))

    for i, radio in enumerate(radiographs):
        for j, tooth in enumerate(radio.teeth):
            if polar:
                data = to_polar(tooth.pr_landmarks)
            else:
                data = tooth.pr_landmarks
            teeth_data[0, j, i, :] = data[0]
            teeth_data[1, j, i, :] = data[1]
            # teeth_data[j, i, :] = tooth.pr_landmarks.reshape(80)

    for j in range(8):
        mean_tooth_data = np.zeros((2, 40))
        evecs = np.zeros((2, 5, 40))
        evals = np.zeros((2, 5))
        cov_mat = np.zeros((2, 40, 40))
        for i in range(2):

            mean_tooth_data[i, :] = teeth_data[i, j].mean(axis=0)

            # trick to expand mean to match dimensions of matrix
            teeth_data[i, j] = teeth_data[i, j] - mean_tooth_data[i][
                                                  np.newaxis, :]

            cov_mat = np.cov(teeth_data[i, j].T)
            evals, evecs = np.linalg.eig(cov_mat)
            idx = np.argsort(evals)[::-1]
            evecs[i] = np.float64(evecs[:, idx])
            evals[i] = np.float64(evals[idx])
            # print(evals.sum())
            print(evals[:5] / evals.sum())
            # print("["+",".join(["{:.2}".format(k) for k in evecs[0]])+"]")

        mean_tooth = Tooth(to_cart(mean_tooth_data))
        teeth_var = np.ones((6,2))
        for i in range(3):
            for j in range(2):
                new_tooth = np.copy(mean_tooth_data)
                new_tooth[j] = new_tooth[j] + evecs[j]
                teeth_var[2 * i, j] = Tooth(new_tooth).to_img()
                new_tooth[j] = new_tooth[j] - 2 * evecs[j]
                teeth_var[2 * i + 1, j] = Tooth(new_tooth).to_img()

