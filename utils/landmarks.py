import math

import numpy as np


def procrustes_transform(landmarks):
    pr_landmarks = np.copy(landmarks)
    pr_landmarks = translate_norm(pr_landmarks)
    pr_landmarks = scale_norm(pr_landmarks)
    pr_landmarks = rotate_norm(pr_landmarks)
    return pr_landmarks


def compute_centroid(landmarks):
    return landmarks.mean(axis=1)


def translate_norm(landmarks):
    centroid = compute_centroid(landmarks)
    landmarks = translate(landmarks, -centroid)
    return landmarks


def scale_norm(landmarks):
    sqrd = landmarks ** 2
    scale_factor = sqrd.sum()
    scale_factor = math.sqrt(scale_factor)

    landmarks = scale(landmarks, scale_factor)
    return landmarks


def rotate_norm(landmarks):
    cov_mat = np.cov(landmarks)
    evals, evecs = np.linalg.eig(cov_mat)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    ref = np.array([0, 1])
    cosR = evecs[0].dot(ref.T)
    sinR = (np.linalg.det(np.stack((evecs[0], ref))) /
            (np.linalg.norm(evecs[0]) * np.linalg.norm(ref)))
    matRot = np.array([[cosR, sinR], [-sinR, cosR]])
    # matRot = np.array([cosR, -sinR], [sinR, cosR])
    landmarks = matRot.dot(landmarks)
    return landmarks


def translate(landmarks, vect):
    landmarks[0] = landmarks[0] + vect[0]
    landmarks[1] = landmarks[1] + vect[1]
    return landmarks


def scale(landmarks, scale_factor):
    landmarks = landmarks / scale_factor
    return landmarks


def rotate(landmarks, angle):
    matRot = np.array([[math.cos(angle), math.sin(angle)],
                       [-math.sin(angle), math.cos(angle)]])
    landmarks = matRot.dot(landmarks.T)
    return landmarks


def procrustes_distance(landmarks1, landmarks2):
    assert landmarks1.shape() == landmarks2.shape()

    dist = landmarks2 - landmarks1
    dist = np.square(dist)
    dist = dist.sum(axis=0)
    dist = np.sqrt(dist)
    dist = dist.sum()
    return dist


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def to_polar(landmarks):
    return np.array([cart2pol(x, y) for x, y in landmarks.T]).T

def to_cart(landmarks):
    return np.array([pol2cart(x, y) for x, y in landmarks.T]).T
