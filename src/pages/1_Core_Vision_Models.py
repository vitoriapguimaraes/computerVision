import cv2
import av
import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image, ImageOps

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from utils.ui import (
    configure_page,
    render_sidebar_info,
    render_instructions_tab
)
from settings.loader import load_cifar10_model, get_cifar10_class_names
from settings.config import IMG_CIFAR10_CLASSES, DEMO_CLASSIFICATION, IMG_CIFAR10_EXAMPLES, MODEL_YOLO

configure_page("Core Vision Models", "🧠")
render_sidebar_info()

st.title("Core Vision Models")
st.markdown("Compare foundational image classification (CIFAR-10) with state-of-the-art real-time object detection (YOLOv8).")

@st.cache_resource(show_spinner="Loading YOLOv8 Model...")
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO(MODEL_YOLO)

yolo_model = load_yolo_model()


tab_demo, tab_example_cifar, tab_execution_cifar, tab_execution_yolo = st.tabs([
    "Instructions & Demos", 
    "CIFAR-10 Examples", 
    "CIFAR-10 Execution", 
    "YOLOv8 Execution"
])

with tab_demo:
    st.markdown("### 1. Image Classification (CIFAR-10)")
    how_it_works_cifar = """
This module uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 distinct categories.
The network analyzes the visual features of the input and returns a confidence score for each possible class.

Upload an image or use your camera to classify it into one of 10 categories using a Convolutional Neural Network.
    """
        
    render_instructions_tab(how_it_works_cifar, DEMO_CLASSIFICATION)

    st.divider()

    st.markdown("### 2. Object Detection (YOLOv8)")
    how_it_works_yolo = """
This module uses **YOLOv8** (You Only Look Once), a state-of-the-art, real-time object detection system by Ultralytics.
The model processes each frame of your webcam feed and draws bounding boxes around 80 different classes of objects (people, cars, cups, cell phones, etc).

> **Note:** The model is optimized to run on the CPU (using the lightweight 'Nano' architecture).
    """
    render_instructions_tab(how_it_works_yolo)

with tab_example_cifar:
    st.markdown("### Example Classifications")
    st.markdown(
        "Here are some sample images from our dataset and how the model classifies them."
    )

    model_cifar = load_cifar10_model()
    class_names = get_cifar10_class_names()

    if model_cifar is None:
        st.error("Model not found. Please train and save it first.")
    else:
        cols = st.columns(3)
        for i, img_path in enumerate(IMG_CIFAR10_EXAMPLES):
            with cols[i]:
                import os
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_display = ImageOps.fit(
                        img, (400, 300), Image.Resampling.LANCZOS
                    )
                    st.image(img_display, use_column_width=True)

                    img_resized = img.resize((32, 32))
                    if img_resized.mode != "RGB":
                        img_resized = img_resized.convert("RGB")
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = model_cifar.predict(img_array, verbose=0)[0]
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

with tab_execution_cifar:
    
    st.image(Image.open(IMG_CIFAR10_CLASSES), use_column_width=True)

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

    model_cifar = load_cifar10_model()

    if model_cifar is None:
        st.error(
            "Error: The pre-trained Keras model (cifar10_cnn.h5) was not found. Please train and save the model first."
        )
    elif image_file is not None:
        image = Image.open(image_file)
        col3.image(image, caption="Input Image", use_column_width=True)

        with col4:
            with st.spinner("Analyzing image features..."):
                img_resized = image.resize((32, 32))
                if img_resized.mode != "RGB":
                    img_resized = img_resized.convert("RGB")

                image_array = np.array(img_resized) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model_cifar.predict(image_array)[0]
                class_names = get_cifar10_class_names()

                predicted_class_idx = np.argmax(predictions)
                predicted_name = class_names[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100

                st.success(
                    f"**Prediction:** {predicted_name} ({confidence:.1f}% confidence)"
                )

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

with tab_execution_yolo:
    st.markdown("### Input Feed")
    col1, col2 = st.columns([1, 2])

    with col1:
        input_method_yolo = st.radio(
            "Select Input Method:", ["Upload Image", "Camera Picture", "Live Video Stream"], horizontal=True, key="yolo_input"
        )
        
    if input_method_yolo == "Live Video Stream":
        st.info("The video is streamed directly from your browser to the local backend via WebRTC.")
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        class YoloVideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = yolo_model(img, verbose=False)
                res_plotted = results[0].plot()
                return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

        webrtc_streamer(
            key="yolov8_detection",
            video_processor_factory=YoloVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        with col2:
            image_file_yolo = None
            if input_method_yolo == "Upload Image":
                image_file_yolo = st.file_uploader(
                    "Choose an image...", type=["jpg", "jpeg", "png"], key="yolo_upload"
                )
            else:
                image_file_yolo = st.camera_input("Take a picture", key="yolo_camera")
        
        st.markdown("### Analysis Results")
        if image_file_yolo is not None:
            image = Image.open(image_file_yolo)
            
            with st.spinner("Detecting objects..."):
                img_array = np.array(image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif len(img_array.shape) == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
                results = yolo_model(img_array, verbose=False)
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, caption="Detected Objects", use_column_width=True)
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"**Found {len(boxes)} object(s)!**")
                else:
                    st.warning("No objects detected in the image.")
        else:
            st.caption("Awaiting input data...")
