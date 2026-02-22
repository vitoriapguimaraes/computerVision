import matplotlib.pyplot as plt
from tensorflow.keras import datasets
import os
import numpy as np


def generate_cifar10_grid():
    output_path = os.path.join("assets", "cifar10_classes_1_line.jpg")
    if os.path.exists(output_path):
        print("Image already exists!")
        return

    print("Loading CIFAR-10...")
    (x_train, y_train), _ = datasets.cifar10.load_data()

    class_names = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    fig, axes = plt.subplots(1, 10, figsize=(12, 3))
    axes = axes.flatten()

    for i in range(10):
        # Find first image of class i
        idx = np.where(y_train == i)[0][0]
        img = x_train[idx]

        axes[i].imshow(img)
        axes[i].set_title(class_names[i], fontsize=14, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    generate_cifar10_grid()
