import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
from PIL import ImageOps
from utils.ui import (
    configure_page,
    render_sidebar_info,
    get_image_base64,
)
from utils.loader import load_cifar10_model, get_cifar10_class_names
from utils.config import IMG_CIFAR10_CLASSES, DEMO_CLASSIFICATION, IMG_CIFAR10_EXAMPLES

configure_page("Image Classification", "🖼️")
render_sidebar_info()

st.title("🖼️ Image Classification (CIFAR-10)")

tab1, tab2, tab3 = st.tabs(["Instructions & Demo", "Example Results", "Execution"])

with tab1:

    st.markdown(
        """
        This module uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 distinct categories.
        The network analyzes the visual features of the input and returns a confidence score for each possible class.
        """
    )

    st.image(Image.open(IMG_CIFAR10_CLASSES), use_column_width=True)

    col1, col2 = st.columns(2)

    col1.markdown("### How it works")
    col1.markdown(
        "Upload an image or use your camera to classify it into one of 10 categories using a Convolutional Neural Network."
    )

    with col2:
        st.markdown("### Demo")

        gif_b64 = get_image_base64(DEMO_CLASSIFICATION)
        st.markdown(
            f'<img src="{gif_b64}" width="100%" style="border-radius: 8px;">',
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown("### Example Classifications")
    st.markdown(
        "Here are some sample images from our dataset and how the model classifies them."
    )

    model = load_cifar10_model()
    class_names = get_cifar10_class_names()

    if model is None:
        st.error("Model not found. Please train and save it first.")
    else:
        cols = st.columns(3)
        for i, img_path in enumerate(IMG_CIFAR10_EXAMPLES):
            with cols[i]:
                import os

                if os.path.exists(img_path):
                    img = Image.open(img_path)

                    # Ensure consistent display size (crop to 400x300)
                    img_display = ImageOps.fit(
                        img, (400, 300), Image.Resampling.LANCZOS
                    )
                    st.image(img_display, use_column_width=True)

                    # Pre-process
                    img_resized = img.resize((32, 32))
                    if img_resized.mode != "RGB":
                        img_resized = img_resized.convert("RGB")
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    preds = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(preds)
                    confidence = preds[pred_idx] * 100

                    st.success(f"**{class_names[pred_idx]}** ({confidence:.1f}%)")

                    import pandas as pd

                    df = pd.DataFrame(
                        {"Category": class_names, "Probability": preds * 100}
                    )
                    df = df.sort_values(by="Probability", ascending=True)

                    fig = px.bar(
                        df,
                        x="Probability",
                        y="Category",
                        orientation="h",
                        title="Confidence Breakdown",
                        color="Probability",
                        color_continuous_scale="Viridis",
                        range_x=[0, 100],
                    )

                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"File not found: {img_path}")

with tab3:

    st.markdown("### Input Feed")

    col1, col2 = st.columns([1, 2])

    with col1:
        input_method = st.radio(
            "Select Input Method:", ["Upload Image", "Use Camera"], horizontal=True
        )
    with col2:
        image_file = None
        if input_method == "Upload Image":
            image_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )
        else:
            image_file = st.camera_input("Take a picture")

    st.markdown("### Analysis Results")

    col3, col4 = st.columns([2, 1])

    model = load_cifar10_model()

    if model is None:
        st.error(
            "Error: The pre-trained Keras model (cifar10_cnn.h5) was not found. Please train and save the model first."
        )
    elif image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)

        col3.image(image, caption="Input Image", use_column_width=True)

        with col4:
            # Pre-process image for CIFAR-10 model
            with st.spinner("Analyzing image features..."):
                img_resized = image.resize((32, 32))

                # Ensure image is RGB
                if img_resized.mode != "RGB":
                    img_resized = img_resized.convert("RGB")

                image_array = np.array(img_resized) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Run inference
                predictions = model.predict(image_array)[0]
                class_names = get_cifar10_class_names()

                # Get top prediction
                predicted_class_idx = np.argmax(predictions)
                predicted_name = class_names[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100

                st.success(
                    f"**Prediction:** {predicted_name} ({confidence:.1f}% confidence)"
                )

                # Plotly Horizontal Bar Chart
                import pandas as pd

                df = pd.DataFrame(
                    {"Category": class_names, "Probability": predictions * 100}
                )
                df = df.sort_values(by="Probability", ascending=True)

                fig = px.bar(
                    df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    title="Confidence Breakdown",
                    color="Probability",
                    color_continuous_scale="Viridis",
                    range_x=[0, 100],
                )

                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.caption("Awaiting input data...")
