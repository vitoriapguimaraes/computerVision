import cv2
import os


from utils.traffic import get_subtractor, apply_filter


def generate_algorithm_comparison(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    algorithms = ["MOG2", "KNN", "GMG", "CNT", "MOG"]
    subtractors = {algo: get_subtractor(algo) for algo in algorithms}

    # Read first 50 frames to train the background subtractors
    frame = None
    for _ in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        for algo in algorithms:
            subtractors[algo].apply(frame)

    if frame is None:
        print("Failed to read frames.")
        cap.release()
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save original frame
    cv2.imwrite(os.path.join(output_dir, "original_frame.jpg"), frame)

    # Save masks
    for algo in algorithms:
        mask = subtractors[algo].apply(frame)
        mask_filtered = apply_filter(mask, "combine")
        cv2.imwrite(os.path.join(output_dir, f"{algo}_mask.jpg"), mask_filtered)

    print(f"Saved comparison images to {output_dir}")
    cap.release()


if __name__ == "__main__":
    generate_algorithm_comparison("assets/Ponte.mp4", "assets/traffic_algo_comparisons")
