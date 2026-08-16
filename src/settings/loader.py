import streamlit as st
import os
from settings.config import MODEL_CIFAR10
@st.cache_resource(show_spinner="Loading Keras Model...")
def load_cifar10_model():
    """Load the pre-trained CIFAR-10 classification model."""
    if not os.path.exists(MODEL_CIFAR10):
        return None

    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_CIFAR10)


def get_cifar10_class_names():
    return [
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
