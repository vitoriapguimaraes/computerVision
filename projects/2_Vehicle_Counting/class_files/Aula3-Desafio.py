import cv2
import sys
import csv
import numpy as np
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
fp = open(os.path.join(output_dir, "Results.csv"), mode="w")
writer = csv.DictWriter(fp, fieldnames=["Frames", "Pixel Count"])
writer.writeheader()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "dataset", "Ponte.mp4")

# KNN = 5.06
# GMG = 11.04
# CNT = 3.53
# MOG = 7.49
# MOG2 = 4.93

algorithm_types = ["KNN", "GMG", "CNT", "MOG", "MOG2"]


def Subtractor(algorithm_type):
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
    print("Detector inválido")
    sys.exit(1)


cap = cv2.VideoCapture(VIDEO)
# e1 = cv2.getTickCount()

background_subtractor = []

for i, a in enumerate(algorithm_types):
    print(i, a)
    background_subtractor.append(Subtractor(a))


def main():
    # frame_number = -1
    while cap.isOpened:
        ok, frame = cap.read()

        if not ok:
            print("Frames acabaram!")
            break

        # frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

        knn = background_subtractor[0].apply(frame)
        gmg = background_subtractor[1].apply(frame)
        cnt = background_subtractor[2].apply(frame)
        mog = background_subtractor[3].apply(frame)
        mog2 = background_subtractor[4].apply(frame)

        knnCount = np.count_nonzero(knn)
        gmgCount = np.count_nonzero(gmg)
        cntCount = np.count_nonzero(cnt)
        mogCount = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)

        writer.writerow({"Frames": "KNN", "Pixel Count": knnCount})
        writer.writerow({"Frames": "GMG", "Pixel Count": gmgCount})
        writer.writerow({"Frames": "CNT", "Pixel Count": cntCount})
        writer.writerow({"Frames": "MOG", "Pixel Count": mogCount})
        writer.writerow({"Frames": "MOG2", "Pixel Count": mog2Count})

        cv2.imshow("Original", frame)
        cv2.imshow("KNN", knn)
        cv2.imshow("GMG", gmg)
        cv2.imshow("CNT", cnt)
        cv2.imshow("MOG", mog)
        cv2.imshow("MOG2", mog2)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            # or frame_number > 300
            break

        # e2 = cv2.getTickCount()
        # t = (e2 - e1) / cv2.getTickFrequency()
        # print(t)


main()
