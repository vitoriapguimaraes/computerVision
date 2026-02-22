import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "dataset", "Arco.mp4")

cap = cv2.VideoCapture(VIDEO)
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "median_frame_cinza.jpg"), grayMedianFrame)

while True:
    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Acabou os frames")
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", frameGray)
    if cv2.waitKey(1) & 0xFF == ord("c"):
        break

cap.release()
