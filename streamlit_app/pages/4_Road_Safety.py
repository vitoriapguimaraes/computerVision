import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from utils.ui import configure_page, render_sidebar_info, get_image_base64
from utils.config import DEMO_DROWSINESS

# --- MediaPipe setup (module-level) ---
_mp_face_mesh = mp.solutions.face_mesh
_mp_draw = mp.solutions.drawing_utils
_mp_draw_styles = mp.solutions.drawing_styles

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
configure_page("CV Hub | Road Safety", "💤")
render_sidebar_info()

st.title("💤 Road Safety (Fatigue Detection)")
st.markdown(
    "Driver safety system tracking Eye Aspect Ratio (EAR) to prevent microsleep events."
)

tab1, tab2 = st.tabs(["Instructions & Demo", "Execution"])

with tab1:
    col1, col2 = st.columns(2)
    col1.markdown("### How it works")
    col1.markdown(
        """
        This module monitors driver fatigue using **MediaPipe FaceMesh** to detect facial landmarks.
        It calculates two key metrics in real-time:

        - **EAR (Eye Aspect Ratio):** Measures eye openness. When EAR drops below a threshold
          for a sustained period, the system triggers a fatigue alert.
        - **MAR (Mouth Aspect Ratio):** Detects yawning by measuring mouth openness.
        - **Blink Counter:** Tracks blink frequency as an additional fatigue indicator.
        """
    )

    with col2:
        st.markdown("### Demo")
        gif_b64 = get_image_base64(DEMO_DROWSINESS)
        st.markdown(
            f'<img src="{gif_b64}" width="100%" style="border-radius: 8px;">',
            unsafe_allow_html=True,
        )

    st.markdown("### EAR & MAR Formulas")

    st.markdown("**Eye Aspect Ratio (EAR)**")
    col3, col4 = st.columns(2)
    col3.code("EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)")
    col4.caption(
        "Where p1–p6 are the 6 eye landmark coordinates. EAR ≈ 0.3 when open, drops to 0 when fully closed."
    )

    st.markdown("**Mouth Aspect Ratio (MAR)**")
    col5, col6 = st.columns(2)
    col5.code("MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|)")
    col6.caption(
        "Where p1–p8 are the 8 mouth contour landmarks. High MAR indicates a yawn."
    )

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

    start_btn = st.button("▶️ Start Camera", type="primary", use_container_width=True)

    st.markdown("### Analysis Results")
    col3, col4 = st.columns([2, 1])

    with col3:
        video_placeholder = st.empty()

    with col4:
        st.markdown("### Live Metrics")
        ear_metric = st.empty()
        mar_metric = st.empty()
        blink_metric = st.empty()
        status_metric = st.empty()

        ear_metric.metric("EAR", "—")
        mar_metric.metric("MAR", "—")
        blink_metric.metric("Blinks", "0")
        status_metric.metric("Status", "Awaiting Feed")

        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    if start_btn:
        cap = cv2.VideoCapture(0)
        face_mesh = _mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        blink_count, closed_counter, eye_was_closed = 0, 0, False

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ear_val, mar_val, status = 0.0, 0.0, "✅ Alert"

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    (
                        frame,
                        ear_val,
                        mar_val,
                        status,
                        blink_count,
                        closed_counter,
                        eye_was_closed,
                    ) = _process_face(
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
                    )

            video_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True,
            )
            ear_metric.metric("EAR", f"{ear_val:.3f}")
            mar_metric.metric("MAR", f"{mar_val:.3f}")
            blink_metric.metric("Blinks", str(blink_count))
            status_metric.metric("Status", status)

        cap.release()
        face_mesh.close()
        st.rerun()
