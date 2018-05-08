import cv2
import numpy as np

from .display import display_images


def detect_edges(img, sobel=True, k_size=3, scale=3):
    """
    Apply edges detection using convolution filters. Two methods are
    available: Soble filters or Laplacian filters
    :param img: Image
    :type img: ndarray
    :param sobel: set to true to use sobel method instead of laplacian
    :type sobel: bool
    :param scale: scaling parameter for edge detection
    :type scale: float
    :param k_size: Kernel size for sobel method
    :type k_size:
    :return: edges image
    :rtype: ndarray
    """
    if sobel:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k_size, scale=scale)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k_size, scale=scale)
        edges = np.uint8(np.sqrt(sobelx ** 2 + sobely ** 2))
    else:
        edges = cv2.Laplacian(img, cv2.CV_64F, ksize=k_size, scale=scale)
        edges = np.absolute(edges)
        edges = np.uint8(edges)

    return edges


def crop_image(image, h_ratio, w_ratio):
    """
    Crops image to capture only region of interest.
    :param image: Image to crop. This image is not modified by the
    operation.
    :return: Cropped image.
    """
    assert (0 < h_ratio < 1 and 0 < w_ratio < 1)

    h, w, d = image.shape
    top, bottom = int(h * (1 - h_ratio) / 2), int(h * (1 + h_ratio) / 2)
    left, right = int(w * (1 - w_ratio) / 2), int(w * (1 + w_ratio) / 2)
    cropped_img = image[top:bottom, left:right].copy()
    return cropped_img


def intensity_normalisation(img, k_size=200, ratio=1):
    """
    Normalize intensity by removing an average of the neighbourhood intensity
    :param img: Image to normalize
    :type img:
    :param k_size: size of the kernel used for normalization
    :type k_size:
    :param ratio: Importance given to the original value. Must be >1
    :type ratio:
    :return:
    :rtype:
    """
    # sq_k = k_size * k_size
    # center = int((k_size - 1) / 2)
    # kernel = -np.ones((k_size, k_size), np.float32)
    # kernel[center, center] += sq_k * ratio
    # kernel = kernel / ((ratio - 1) * sq_k)
    # np.set_printoptions(threshold=np.nan)
    # image0 = cv2.filter2D(img, -1, kernel)
    blur = cv2.GaussianBlur(img, (155, 155), 155 / 2) / (ratio * 3)
    for k_size in [55, 105]:
        blur += cv2.GaussianBlur(img, (k_size, k_size), k_size / 2) / (
        ratio * 3)
    image0 = (1 + 1 / ratio) * img - blur
    image0[image0 > 255] = 255
    image0 = np.absolute(image0)
    image0 = np.uint8(image0)

    return image0


def process_image(image,
                  median_kernel=5,
                  bilateral_kernel=17,
                  bilateral_color=15,
                  sobel=True,
                  edge_kernel=3,
                  scale_edges=1,
                  threshold=10,
                  crop=False):
    image_p = crop_image(image, 0.8, 0.8) if crop else image
    image_p = cv2.medianBlur(image_p, median_kernel)
    image_p = cv2.bilateralFilter(image_p,
                                  bilateral_kernel,
                                  bilateral_color,
                                  200)
    image_p = detect_edges(image_p,
                           sobel=sobel,
                           k_size=edge_kernel,
                           scale=scale_edges)
    image_p = cv2.GaussianBlur(image_p, (7, 7), 0.5)

    # Super threshold filter
    image_p[image_p < threshold] = 0
    return image_p


def process_image_test(image,
                       median_kernel=5,
                       bilateral_kernel=17,
                       bilateral_color=9):
    """
    Filters image by using median and bilateral filters followed by
    Scharr operator.
    :param image: Image to process. This image is not modified by the
    operation.
    :param median_kernel: The size of median filter kernel.
    :param bilateral_kernel: The size of bilateral filter kernel.
    :param bilateral_color: A color delta that is still considered to
    represent the same color.
    :return: New, processed image.
    """
    image0 = image
    image1 = cv2.medianBlur(image0, median_kernel)
    # NB, last value of bilateralFilter is used only if bilateral_kernel == 0
    # Here it is useless
    image2 = cv2.bilateralFilter(image1, bilateral_kernel, bilateral_color,
                                 200)
    image3 = detect_edges(image2)

    image4 = cv2.GaussianBlur(image3, (7, 7), 1)

    image0_b = intensity_normalisation(image, k_size=65, ratio=2)
    image0_b = intensity_normalisation(image0_b, k_size=25, ratio=2)
    image0_b = intensity_normalisation(image0_b, k_size=5, ratio=2)
    image1_b = cv2.GaussianBlur(image0_b, (7, 7), 0)
    image1_b = cv2.medianBlur(image1_b, median_kernel)
    # NB, last value of bilateralFilter is used only if bilateral_kernel == 0
    # Here it is useless
    image2_b = cv2.bilateralFilter(image1_b,
                                   35,
                                   bilateral_color,
                                   200)
    image2_b = cv2.GaussianBlur(image2_b, (5, 5), 0.5)

    image3_b = detect_edges(image2_b,scale=0.05,k_size=5)
    image3_b[image3_b < 30] = 0
    image4_b = cv2.GaussianBlur(image3_b, (7, 7), 1)

    display_images([[image, image0, image2, image3, image4],
                    [image, image0_b, image2_b, image3_b, image4_b]])

    return
