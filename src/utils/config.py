from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
DEMOS_DIR = ASSETS_DIR / "demos"

# Dashboard Previews
IMG_PREVIEW_CLASSIFICATION = str(
    DEMOS_DIR / "Image_Classification_CIFAR10_display_preview.jpg"
)
IMG_PREVIEW_TRAFFIC = str(DEMOS_DIR / "Vehicle_Counting_display_preview.jpg")
IMG_PREVIEW_TRACKING = str(DEMOS_DIR / "Hand_Tracking_display_preview.jpg")
IMG_PREVIEW_DROWSINESS = str(
    DEMOS_DIR / "Driver_Drowsiness_Detection_display_preview.jpg"
)

# Demos (GIFs)
DEMO_CLASSIFICATION = str(DEMOS_DIR / "Image_Classification_CIFAR10_display_opt.gif")
DEMO_TRAFFIC = str(DEMOS_DIR / "Vehicle_Counting_display_opt.gif")
DEMO_TRACKING = str(DEMOS_DIR / "Hand_Tracking_display_opt.gif")
DEMO_DROWSINESS = str(DEMOS_DIR / "Driver_Drowsiness_Detection_display_opt.gif")

# Image Classification specific
IMG_CIFAR10_CLASSES = str(ASSETS_DIR / "cifar10_classes_1_line.jpg")
IMG_CIFAR10_EXAMPLES = [
    str(ASSETS_DIR / "examples_image_classification" / "image-entry.jpg"),
    str(ASSETS_DIR / "examples_image_classification" / "image-entry1.jpg"),
    str(ASSETS_DIR / "examples_image_classification" / "image-entry2.jpg"),
]

# Traffic Analysis specific
TRAFFIC_ALGO_COMPARISONS_DIR = ASSETS_DIR / "traffic_algo_comparisons"
IMG_TRAFFIC_ORIGINAL = str(TRAFFIC_ALGO_COMPARISONS_DIR / "original_frame.jpg")


def get_traffic_mask_path(algo: str) -> str:
    return str(TRAFFIC_ALGO_COMPARISONS_DIR / f"{algo}_mask.jpg")


VIDEO_TRAFFIC_DEFAULT = str(ASSETS_DIR / "Ponte.mp4")

# Models
MODEL_CIFAR10 = str(MODELS_DIR / "cifar10_cnn.h5")
