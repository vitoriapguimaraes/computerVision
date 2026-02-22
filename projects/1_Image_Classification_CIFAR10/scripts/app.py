# Projeto de Classificação de Imagens com Redes Neurais Convolucionais (CNNs) usando TensorFlow

"""
(1) Importação das bibliotecas necessárias

Importa bibliotecas para manipulação de arrays, imagens, visualização e construção de modelos de deep learning
platform: para checar a versão do Python
tensorflow: para criar e treinar o modelo de classificação
matplotlib: para visualização de imagens
numpy: para manipulação de arrays
PIL: para abrir e processar imagens externas
"""
from platform import python_version
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def print_versions():
    """Exibe as versões do Python e TensorFlow."""
    print(f"Python version: {python_version()}")
    print(f"TensorFlow version: {tf.__version__}")


def load_data():
    """Carrega o dataset CIFAR-10 e retorna os dados de treino e teste."""
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def normalize_images(images):
    """Normaliza os valores dos pixels para o intervalo [0, 1]."""
    return images / 255.0


def visualize_images(images, labels, class_names):
    """Exibe uma grade 5x5 com as imagens e seus respectivos rótulos."""
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i][0]])
    plt.show()


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


def train_model(model, x_train, y_train, x_val, y_val, epochs=10):
    """Compila e treina o modelo, retornando o histórico do treinamento."""
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    print("Training the model...")
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    print("Training the model...ok")
    return history


def evaluate_model(model, x_test, y_test):
    """Avalia o modelo e imprime a acurácia nos dados de teste."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Accuracy with Test Data: {accuracy}")
    return loss, accuracy


def predict_image(model, image_path, class_names):
    """Classifica uma imagem externa e imprime o resultado."""
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    predicted_name = class_names[predicted_class]
    print(f"The input image ({image_path}) was classified as '{predicted_name}'.")
    return predicted_name


def main():
    print_versions()
    print("(1) Importing the necessary libraries - Status: done\n")

    print("(2) Loading and initial exploration of data - Status: done\n")
    (images_training, labels_training), (images_testing, labels_testing) = load_data()
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

    print("(3) Image pre-processing - Status: done\n")
    images_training = normalize_images(images_training)
    images_testing = normalize_images(images_testing)
    visualize_images(images_training, labels_training, classification_names)

    import os

    # Determina dinamicamente o diretório raiz do projeto (uma pasta acima da pasta scripts)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cifar10_cnn_v2.h5")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = models.load_model(model_path)
        print(
            "(4) Building and Training the Model - Status: skipped (loaded existing model)\n"
        )
    else:
        print("(4) Building and Training the Model - Status: done\n")
        model = build_model((32, 32, 3), len(classification_names))
        print("### Classification model Summary ###")
        print(model.summary())
        train_model(
            model, images_training, labels_training, images_testing, labels_testing
        )

        print("Saving the trained model to h5 file...")
        model.save(model_path)
        print(f"Model saved at {model_path}")

    print("(5) Performance evaluation - Status: done\n")
    evaluate_model(model, images_testing, labels_testing)

    print("(6) Model deployment and predictions - Status: done\n")
    images_dir = os.path.join(BASE_DIR, "images")

    if os.path.exists(images_dir):
        for img_name in os.listdir(images_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                data_path = os.path.join(images_dir, img_name)
                print(f"Imported {data_path}")
                predict_image(model, data_path, classification_names)
    else:
        print(
            f"Directory {images_dir} not found. Please create it and add images to test."
        )


if __name__ == "__main__":
    main()
