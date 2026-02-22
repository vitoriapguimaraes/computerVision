import os
from PIL import Image


def generate_previews(assets_dir="assets"):
    """
    Looks for .gif files in the specified directory and generates
    lightweight JPEG preview images from their first frame.
    These previews are used in the Streamlit dashboard to prevent browser freezing.
    """
    if not os.path.exists(assets_dir):
        print(f"Directory {assets_dir} not found.")
        return

    count = 0
    for filename in os.listdir(assets_dir):
        if filename.endswith(".gif"):
            gif_path = os.path.join(assets_dir, filename)
            preview_filename = filename.replace(".gif", "_preview.jpg")
            preview_path = os.path.join(assets_dir, preview_filename)

            # Skip if preview already exists
            if os.path.exists(preview_path):
                print(f"Skipping {filename} (preview already exists)")
                continue

            try:
                print(f"Generating preview for {filename}...")
                img = Image.open(gif_path)
                # Convert to RGB to discard alpha channel from gif if any, required for JPEG
                img_rgb = img.convert("RGB")
                # Save as JPEG with optimized quality
                img_rgb.save(preview_path, "JPEG", quality=85)
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nDone! Generated {count} preview(s).")


if __name__ == "__main__":
    # Assuming this script is run from the project root or inside utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    assets_folder = os.path.join(project_root, "assets")

    generate_previews(assets_folder)
