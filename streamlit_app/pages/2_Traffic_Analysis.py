import streamlit as st
import cv2
from PIL import Image
from utils.ui import configure_page, render_sidebar_info, get_image_base64
from utils.traffic import get_subtractor, apply_filter, get_centroid, save_uploaded_file
from utils.config import (
    DEMO_TRAFFIC,
    IMG_TRAFFIC_ORIGINAL,
    get_traffic_mask_path,
    VIDEO_TRAFFIC_DEFAULT,
)

configure_page("Traffic Analysis", "🚗")
render_sidebar_info()

st.title("🚗 Traffic Analysis (Vehicle Counting)")
st.markdown(
    "Automated traffic flow monitoring and vehicle counting using OpenCV background subtraction."
)

tab1, tab2, tab3 = st.tabs(["Instructions & Demo", "Algorithm Comparison", "Execution"])

with tab1:
    st.markdown(
        "This module uses OpenCV background subtraction (MOG2) to detect moving vehicles in a video stream. It draws bounding boxes around detected objects and counts them as they cross a defined virtual line."
    )

    col1, col2 = st.columns(2)
    col1.markdown("### How it works")
    col1.markdown(
        "The system subtracts the background of a static video feed. Contours larger than a specific area are considered vehicles. Once their center crosses the imaginary line, the counter increments."
    )

    with col2:
        st.markdown("### Demo")
        gif_b64 = get_image_base64(DEMO_TRAFFIC)
        st.markdown(
            f'<img src="{gif_b64}" width="100%" style="border-radius: 8px;">',
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown("### Algorithm Comparison")
    st.markdown(
        "Different background subtraction algorithms perform differently based on lighting, shadows, and object speed. "
        "Here is a side-by-side comparison of the 5 implemented algorithms applied to the same frame after learning the background."
    )

    col1, col2 = st.columns([1, 2])
    col1.markdown("#### Frame 50")
    col1.image(Image.open(IMG_TRAFFIC_ORIGINAL), use_column_width=True)

    col2.markdown("#### Detected Masks")
    mask_cols = col2.columns(5)
    algorithms = ["MOG2", "KNN", "GMG", "CNT", "MOG"]

    for i, algo in enumerate(algorithms):
        with mask_cols[i]:
            st.image(
                Image.open(get_traffic_mask_path(algo)),
                caption=algo,
                use_column_width=True,
                clamp=True,
            )


with tab3:
    st.markdown("### Input Feed")

    col1, col2 = st.columns([1, 2])

    with col1:
        input_method = st.radio(
            "Select Input Method:", ["Default Video", "Upload Video"], horizontal=True
        )

        algorithm_type = st.selectbox(
            "Select Algorithm:", ["MOG2", "KNN", "GMG", "CNT", "MOG"]
        )

    with col2:
        video_file = None
        if input_method == "Upload Video":
            video_file_buffer = st.file_uploader(
                "Choose a video...", type=["mp4", "avi", "mov"]
            )
            if video_file_buffer is not None:
                video_file = save_uploaded_file(video_file_buffer)
        else:
            video_file = VIDEO_TRAFFIC_DEFAULT
            import os

            filename = os.path.basename(VIDEO_TRAFFIC_DEFAULT)
            st.info(f"Using default dataset video ({filename}).")

        start_btn = st.button(
            "▶️ Start Analysis", type="primary", use_container_width=True
        )

    st.markdown("### Analysis Results")

    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown("#### Video Feed")
        video_placeholder = st.empty()

    with col4:
        st.markdown("### Real-Time Metrics")
        mcol1, mcol2 = st.columns(2)
        veh_metric = mcol1.empty()
        algo_metric = mcol2.empty()

        veh_metric.metric("Total Vehicles", "0")
        algo_metric.metric("Algorithm", algorithm_type, None)

        stop_btn = st.button("⏹️ Stop/Reset", use_container_width=True)

    if start_btn and video_file:
        # Run CV2 Loop
        cap = cv2.VideoCapture(video_file)
        background_subtractor = get_subtractor(algorithm_type)

        MIN_WIDTH = 40
        MIN_HEIGHT = 40
        PIXEL_OFFSET = 2
        ROI_LINE = 620
        vehicle_count = 0
        centroids = []

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                veh_metric.metric("Total Vehicles", str(vehicle_count))
                break

            # Process Frame
            mask = background_subtractor.apply(frame)
            mask = apply_filter(mask, "combine")

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.line(frame, (25, ROI_LINE), (1200, ROI_LINE), (255, 127, 0), 3)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    centroid = get_centroid(x, y, w, h)
                    centroids.append(centroid)
                    cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

            for x, y in centroids[:]:
                if (ROI_LINE + PIXEL_OFFSET) > y > (ROI_LINE - PIXEL_OFFSET):
                    vehicle_count += 1
                    cv2.line(frame, (25, ROI_LINE), (1200, ROI_LINE), (0, 127, 255), 3)
                    centroids.remove((x, y))

            # Update UI
            # Convert frame BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            veh_metric.metric("Total Vehicles", str(vehicle_count))

        cap.release()
