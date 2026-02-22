import numpy as np
import cv2
from time import sleep
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "dataset", "Rua.mp4")
delay = 10

cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# print(medianFrame)
# cv2.imshow('Median frame', medianFrame)
# cv2.waitKey(0)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "median_frame.jpg"), medianFrame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Cinza', grayMedianFrame)
# cv2.waitKey(0)
cv2.imwrite(os.path.join(output_dir, "gray_median_frame.jpg"), grayMedianFrame)

while True:
    tempo = float(1 / delay)
    sleep(tempo)

    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Acabou os frames")
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("frames", dframe)
    if cv2.waitKey(1) & 0xFF == ord("c"):
        break

cap.release()
