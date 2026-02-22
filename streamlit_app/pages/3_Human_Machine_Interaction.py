import streamlit as st
from utils.ui import configure_page, render_sidebar_info, get_image_base64
from utils.config import DEMO_TRACKING

configure_page("Gesture Tracking", "🤚")
render_sidebar_info()

st.title("🤚 Human-Machine Interaction (Hand Tracking)")
st.markdown(
    "Real-time hand landmark detection using MediaPipe for touchless interfaces."
)

tab1, tab2 = st.tabs(["Instructions & Demo", "Execution"])

with tab1:
    st.markdown(
        "This module uses Google's MediaPipe framework to detect hand landmarks in real-time. It maps 21 3D coordinates across the hand, allowing for complex gesture recognition and touchless interfaces."
    )

    col1, col2 = st.columns(2)
    col1.markdown("### How it works")
    col1.markdown(
        "The MediaPipe Hands solution utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box."
    )
    col1.markdown(
        "A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints."
    )

    with col2:
        st.markdown("### Demo")
        gif_b64 = get_image_base64(DEMO_TRACKING)
        st.markdown(
            f'<img src="{gif_b64}" width="100%" style="border-radius: 8px;">',
            unsafe_allow_html=True,
        )

    st.markdown("### Gestures & Commands (Local Script)")
    st.markdown(
        """
        - **Type text:** Use right hand. Touch virtual keys with index finger to type. To erase, raise only the right pinky.
        - **Open apps (left hand):** Index up opens Word. Index + middle up opens Excel. Index + middle + ring up opens Firefox. All fingers down closes Firefox.
        - **Draw (two hands):** Left hand sets brush color (1 up: blue, 2 up: green, 3 up: red, 4 up: eraser). Right hand draws with index finger. Right hand distance to camera controls brush thickness.
        """
    )

with tab2:
    st.markdown("### Input Feed")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.info("Uses your local webcam (Index 0).")

    with col2:
        start_btn = st.button(
            "▶️ Start Camera", type="primary", use_container_width=True
        )

    st.markdown("### Analysis Results")

    col3, col4 = st.columns([2, 1])

    with col3:
        video_placeholder = st.empty()

    with col4:
        st.markdown("### Live Metrics")
        hands_metric = st.empty()
        gest_metric = st.empty()

        hands_metric.metric("Hands Detected", "0")
        gest_metric.metric("Active Finger", "None", None)

        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    if start_btn:
        import cv2
        from utils.hand_tracking import HandTracker

        cap = cv2.VideoCapture(0)
        tracker = HandTracker()

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror image
            frame, all_hands = tracker.find_hands(frame)

            # Extract Metrics
            hands_count = len(all_hands)
            fingers_status = "None"

            if hands_count > 0:
                fingers = tracker.fingers_up(all_hands[0])
                up_count = sum(fingers)
                fingers_status = f"{up_count} Fingers Up"

            # Convert frame BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            hands_metric.metric("Hands Detected", str(hands_count))
            gest_metric.metric("Active Gesture", fingers_status, None)

        cap.release()
        st.rerun()
