import cv2
import numpy as np
import os


def get_kernel(kernel_type):
    if kernel_type == "dilation":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if kernel_type in ["opening", "closing"]:
        return np.ones((3, 3), np.uint8)
    return None


def apply_filter(img, filter_type):
    if filter_type == "closing":
        return cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2
        )
    if filter_type == "opening":
        return cv2.morphologyEx(
            img, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2
        )
    if filter_type == "dilation":
        return cv2.dilate(img, get_kernel("dilation"), iterations=2)
    if filter_type == "combine":
        closing = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2
        )
        opening = cv2.morphologyEx(
            closing, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2
        )
        dilation = cv2.dilate(opening, get_kernel("dilation"), iterations=2)
        return dilation
    return img


def get_subtractor(algorithm_type):
    if algorithm_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    return cv2.createBackgroundSubtractorMOG2()


def get_centroid(x, y, w, h):
    """Calculate the centroid of a rectangle."""
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    os.makedirs("tmp", exist_ok=True)
    file_path = os.path.join("tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
