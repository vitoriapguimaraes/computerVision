import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "dataset", "Peixes.mp4")

cap = cv2.VideoCapture(VIDEO)
hasFrame, frame = cap.read()

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
print(medianFrame)
cv2.imshow("Median frame", medianFrame)
cv2.waitKey(0)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "median_frame.jpg"), medianFrame)
