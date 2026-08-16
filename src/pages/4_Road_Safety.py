import cv2
import streamlit as st
import numpy as np
from settings.config import DEMO_DROWSINESS
from utils.ui import configure_page, render_sidebar_info, get_image_base64, render_instructions_tab

configure_page("Road Safety", "🛣️")
render_sidebar_info()
st.title("🛣️ Road Safety (Driver Drowsiness)")
st.markdown("Real-time fatigue monitoring using EAR/MAR tracking.")

@st.cache_resource(show_spinner="Loading MediaPipe FaceMesh Model...")
def load_mediapipe():
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
    _mp_draw = mp.solutions.drawing_utils
    _mp_draw_styles = mp.solutions.drawing_styles
    return mp, _mp_face_mesh, _mp_draw, _mp_draw_styles

mp, _mp_face_mesh, _mp_draw, _mp_draw_styles = load_mediapipe()

_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_MOUTH = [61, 39, 37, 0, 267, 269, 291, 405]


# --- Helper functions ---
def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _compute_ear(lm, eye_ids, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_ids]
    a = _euclidean(pts[1], pts[5])
    b = _euclidean(pts[2], pts[4])
    c = _euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.0


def _compute_mar(lm, mouth_ids, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in mouth_ids]
    a = _euclidean(pts[1], pts[7])
    b = _euclidean(pts[2], pts[6])
    c = _euclidean(pts[3], pts[5])
    d = _euclidean(pts[0], pts[4])
    return (a + b + c) / (2.0 * d) if d > 0 else 0.0


def _update_blink_state(
    ear_val,
    ear_threshold,
    closed_frames_threshold,
    blink_count,
    closed_counter,
    eye_was_closed,
):
    """Returns (status, alert, blink_count, closed_counter, eye_was_closed)."""
    alert = False
    status = "✅ Alert"
    if ear_val < ear_threshold:
        closed_counter += 1
        if not eye_was_closed:
            blink_count += 1
            eye_was_closed = True
        if closed_counter >= closed_frames_threshold:
            alert = True
            status = "⚠️ FATIGUE DETECTED!"
    else:
        closed_counter = 0
        eye_was_closed = False
    return status, alert, blink_count, closed_counter, eye_was_closed


def _draw_overlay(frame, face_lms, ear_val, mar_val, alert):
    """Draw face mesh contours and EAR/MAR text onto frame."""
    # BGR: red on alert, yellow normally
    line_color = (0, 0, 255) if alert else (0, 220, 255)
    custom_spec = _mp_draw.DrawingSpec(color=line_color, thickness=1, circle_radius=1)
    _mp_draw.draw_landmarks(
        frame,
        face_lms,
        _mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=custom_spec,
        connection_drawing_spec=custom_spec,
    )
    cv2.putText(
        frame,
        f"EAR: {ear_val:.2f}  MAR: {mar_val:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        line_color,
        2,
    )
    if alert:
        cv2.putText(
            frame,
            "!!! DROWSINESS ALERT !!!",
            (10, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 0, 255),
            3,
        )


def _process_face(
    frame,
    face_lms,
    ear_threshold,
    mar_threshold,
    closed_frames_threshold,
    w,
    h,
    blink_count,
    closed_counter,
    eye_was_closed,
):
    lm = face_lms.landmark
    left_ear = _compute_ear(lm, _LEFT_EYE, w, h)
    right_ear = _compute_ear(lm, _RIGHT_EYE, w, h)
    ear_val = (left_ear + right_ear) / 2.0
    mar_val = _compute_mar(lm, _MOUTH, w, h)

    status, alert, blink_count, closed_counter, eye_was_closed = _update_blink_state(
        ear_val,
        ear_threshold,
        closed_frames_threshold,
        blink_count,
        closed_counter,
        eye_was_closed,
    )
    if mar_val > mar_threshold:
        status = "🥱 Yawning!"

    _draw_overlay(frame, face_lms, ear_val, mar_val, alert)
    return frame, ear_val, mar_val, status, blink_count, closed_counter, eye_was_closed


# --- Page layout ---

tab1, tab2 = st.tabs(["Instructions & Demo", "Execution"])

with tab1:
    how_it_works = """
This module monitors driver fatigue using **MediaPipe FaceMesh** to detect facial landmarks.
It calculates two key metrics in real-time:

- **EAR (Eye Aspect Ratio):** Measures eye openness. When EAR drops below a threshold
  for a sustained period, the system triggers a fatigue alert.
- **MAR (Mouth Aspect Ratio):** Detects yawning by measuring mouth openness.
- **Blink Counter:** Tracks blink frequency as an additional fatigue indicator.
    """
    
    def render_formulas():
        st.markdown("### EAR & MAR Formulas")

        st.markdown("**Eye Aspect Ratio (EAR)**")
        col3, col4 = st.columns(2)
        col3.code("EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)")
        col4.caption(
            "Where p1–p6 are the 6 eye landmark coordinates. EAR ≈ 0.3 when open, drops to 0 when fully closed."
        )

        st.markdown("**Mouth Aspect Ratio (MAR)**")
        col5, col6 = st.columns(2)
        col5.code("MAR = |p2-p8| / |p1-p5|")
        col6.caption("Where p1–p8 are the mouth landmark coordinates.")
        
    render_instructions_tab(how_it_works, DEMO_DROWSINESS, render_formulas)

with tab2:
    st.markdown("### Input Feed")
    col1, col2, col3 = st.columns(3)

    with col1:
        ear_threshold = st.slider(
            "EAR Alert Threshold",
            min_value=0.1,
            max_value=0.4,
            value=0.25,
            step=0.01,
            help="Eyes below this ratio trigger a fatigue alert.",
        )
    with col2:
        mar_threshold = st.slider(
            "MAR Yawn Threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.55,
            step=0.01,
            help="Mouth above this ratio counts as a yawn.",
        )
    with col3:
        closed_frames_threshold = st.number_input(
            "Closed Frames to Alert",
            min_value=5,
            max_value=60,
            value=20,
            help="Consecutive frames with low EAR before alerting.",
        )


    st.markdown("### Analysis Results")
    col3, col4 = st.columns([2, 1])

    with col3:
        video_placeholder = st.empty()

    with col4:
        st.markdown("### Live Metrics")
        st.info("Metrics are drawn directly on the video feed when using WebRTC.")

    # Inform the user about WebRTC
    st.info("The video is now streamed directly from your browser to the Docker container via WebRTC.")

    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class DrowsinessProcessor:
        def __init__(self):
            self.face_mesh = _mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.blink_count = 0
            self.closed_counter = 0
            self.eye_was_closed = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    (
                        img,
                        ear_val,
                        mar_val,
                        status,
                        self.blink_count,
                        self.closed_counter,
                        self.eye_was_closed,
                    ) = _process_face(
                        img,
                        face_lms,
                        ear_threshold,
                        mar_threshold,
                        closed_frames_threshold,
                        w,
                        h,
                        self.blink_count,
                        self.closed_counter,
                        self.eye_was_closed,
                    )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Start the WebRTC streamer
    webrtc_streamer(
        key="drowsiness_detection",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
