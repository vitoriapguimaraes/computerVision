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

    # Inform the user about WebRTC
    st.info("The video is now streamed directly from your browser to the Docker container via WebRTC.")

    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av
    from utils.hand_tracking import HandTracker
    import cv2

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class HandTrackingProcessor:
        def __init__(self):
            self.tracker = HandTracker()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror image

            # The tracker modifies the image in-place and returns hands data
            img, all_hands = self.tracker.find_hands(img)

            # Convert back to av.VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Start the WebRTC streamer
    webrtc_streamer(
        key="hand_tracking",
        video_processor_factory=HandTrackingProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown(
        """
        > **Note:** Real-time metric extraction from the WebRTC frame directly into the Streamlit UI (outside the video player) requires more complex state management with WebRTC contexts. For now, the visual overlay on the video itself represents the successful processing of the model!
        """
    )
