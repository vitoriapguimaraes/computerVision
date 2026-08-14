import os
from tensorflow.keras import datasets, layers, models


def build_model(input_shape, num_classes):
    """Cria e retorna o modelo CNN para classificação de imagens."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def main():
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    print("Normalizing images...")
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    classification_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    print("Building model...")
    model = build_model((32, 32, 3), len(classification_names))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print("Training model (this will take some time)...")
    # Using fewer epochs for demonstration if needed, but keeping original 10
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {accuracy}")

    # Determine safe output path
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "cifar10_cnn.h5")

    print(f"Saving model to {output_path} ...")
    model.save(output_path)
    print("Optimization complete!")


if __name__ == "__main__":
    main()
