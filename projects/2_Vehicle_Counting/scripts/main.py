import numpy as np
import cv2
import sys
import os

# Determina dinamicamente o diretório raiz do projeto (uma pasta acima da pasta scripts)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "dataset", "Ponte.mp4")

ALGORITHM_TYPES = ["KNN", "GMG", "CNT", "MOG", "MOG2"]
ALGORITHM_TYPE = ALGORITHM_TYPES[1]


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
    print("Invalid detector")
    sys.exit(1)


# Minimum rectangle size for detection
MIN_WIDTH = 40
MIN_HEIGHT = 40
PIXEL_OFFSET = 2
ROI_LINE = 620
vehicle_count = 0


def get_centroid(x, y, w, h):
    """Calculate the centroid of a rectangle."""
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


centroids = []


def update_vehicle_count(centroids, frame):
    global vehicle_count
    for x, y in centroids[:]:
        if (ROI_LINE + PIXEL_OFFSET) > y > (ROI_LINE - PIXEL_OFFSET):
            vehicle_count += 1
            cv2.line(frame, (25, ROI_LINE), (1200, ROI_LINE), (0, 127, 255), 3)
            centroids.remove((x, y))
            print(f"Vehicles detected so far: {vehicle_count}")


def display_info(frame):
    text = f"Vehicles: {vehicle_count}"
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    background_subtractor = get_subtractor(ALGORITHM_TYPE)
    global centroids

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No more frames!")
            break

        mask = background_subtractor.apply(frame)
        mask = apply_filter(mask, "combine")

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame, (25, ROI_LINE), (1200, ROI_LINE), (255, 127, 0), 3)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centroid = get_centroid(x, y, w, h)
                centroids.append(centroid)
                cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

        update_vehicle_count(centroids, frame)
        display_info(frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or 'q'
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
