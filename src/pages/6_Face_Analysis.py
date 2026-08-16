import streamlit as st
import cv2
import av
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from deepface import DeepFace

from utils.ui import configure_page, render_sidebar_info, render_instructions_tab
from settings.config import IMG_PREVIEW_FACE_ANALYSIS
import glob
import os
import numpy as np
from pathlib import Path

configure_page("Face Analysis", "🧑")
render_sidebar_info()

st.title("Real-time Face Analysis")
st.markdown("Detect emotions, age, and gender using DeepFace")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FaceAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = []
        self.lock = threading.Lock()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Analyze every 15 frames (~2 times per second at 30fps)
        if self.frame_count % 15 == 0:
            try:
                # enforce_detection=False prevents crash if no face is found
                results = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], enforce_detection=False, detector_backend='mtcnn', silent=True)
                
                # DeepFace can return a list of dictionaries if multiple faces are detected
                if not isinstance(results, list):
                    results = [results]
                    
                with self.lock:
                    self.last_results = results
            except Exception as e:
                # If analysis fails, we don't update last_results
                print(f"DeepFace error: {e}")
                pass
                
        # Draw the last known results on the current frame
        with self.lock:
            for face in self.last_results:
                # Some deepface versions return region as face['region'], some as face['box']
                # But it's usually 'region'
                if 'region' in face:
                    region = face['region']
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    
                    if w > 0 and h > 0:
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                        
                        # Extract data
                        emotion = face.get('dominant_emotion', 'Unknown')
                        age = face.get('age', '?')
                        
                        # Handle gender dict in newer DeepFace versions
                        gender_data = face.get('gender', {})
                        if isinstance(gender_data, dict):
                            gender = max(gender_data.items(), key=lambda item: item[1])[0] if gender_data else 'Unknown'
                        else:
                            gender = gender_data
                            
                        # Format text
                        info_text = f"{gender}, {age} | {emotion}"
                        
                        # Draw text background
                        (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (255, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(img, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


tab_demo, tab_examples, tab_execution = st.tabs(["Instructions & Demo", "Face Analysis Examples", "Execution"])

with tab_demo:
    how_it_works = """
This module uses the **DeepFace** library to extract facial attributes in real-time from your webcam:
1. **Face Detection:** Locates faces within the video frame.
2. **Attribute Extraction:** Analyzes each detected face for:
   - **Emotion:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
   - **Age:** Estimated age in years
   - **Gender:** Estimated gender

> **Note:** The first time you activate the camera, DeepFace will download the required pre-trained weights.
> To keep the video feed smooth, the analysis is executed on a throttled schedule (not every frame).
    """
    
    render_instructions_tab(how_it_works, IMG_PREVIEW_FACE_ANALYSIS)

with tab_examples:
    st.markdown("### Example Face Analyses")
    st.markdown("Here are some sample images and how DeepFace analyzed them.")
    
    import json
    import pandas as pd
    import plotly.express as px
    from PIL import Image, ImageOps
    
    report_path = Path("src/assets/images_face_analysis/annotated/report.json")
    if report_path.exists():
        with open(report_path, "r") as f:
            report_data = json.load(f)
            
        if report_data:
            cols = st.columns(3)
            for i, (filename, faces) in enumerate(report_data.items()):
                with cols[i % 3]:
                    annotated_path = Path("src/assets/images_face_analysis/annotated") / filename
                    if annotated_path.exists():
                        img_display = Image.open(annotated_path)
                        img_display = ImageOps.fit(img_display, (400, 300), Image.Resampling.LANCZOS)
                        st.image(img_display, use_column_width=True)
                        
                        if faces:
                            face = faces[0]
                            st.success(f"**{face.get('gender', 'Unknown')}**, {face.get('age', '?')} years | {face.get('dominant_emotion', 'Unknown')}")
                            
                            probs = face.get("emotion_probs", {})
                            if probs:
                                df = pd.DataFrame({
                                    "Emotion": list(probs.keys()),
                                    "Probability": list(probs.values())
                                })
                                df = df.sort_values(by="Probability", ascending=True)
                                
                                fig = px.bar(
                                    df,
                                    x="Probability",
                                    y="Emotion",
                                    orientation="h",
                                    title="Emotion Breakdown",
                                    color="Probability",
                                    color_continuous_scale="Viridis",
                                    range_x=[0, 100],
                                )
                                fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Annotated image not found: {filename}")
    else:
        st.warning("Pre-computed report not found. Run the precompute script first.")

with tab_execution:
    st.markdown("### Input Method")
    input_method = st.radio("Select Input Method:", ["Upload Image", "Use WebCam"], horizontal=True)
    
    if input_method == "Use WebCam":
        st.info("Allow camera access and click 'START' to begin real-time face analysis.")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            webrtc_streamer(
                key="face-analysis",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=FaceAnalyzer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            
        with col2:
            st.info("""
            ### About DeepFace
            DeepFace is a lightweight face recognition and facial attribute analysis framework.
            
            In this page, we extract:
            - **Emotion:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
            - **Age:** Estimated age in years
            - **Gender:** Estimated gender
            
            _Processing is throttled to ensure the camera feed remains responsive._
            """)
            
    else:
        image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

        if image is not None:
            # Process static image
            with st.spinner("Analyzing faces..."):
                try:
                    # Use mtcnn and enforce_detection=True for static image
                    results = DeepFace.analyze(image, actions=['emotion', 'age', 'gender'], enforce_detection=True, detector_backend='mtcnn', silent=True)
                    if not isinstance(results, list):
                        results = [results]
                        
                    for face in results:
                        if 'region' in face:
                            region = face['region']
                            x = region.get('x', 0)
                            y = region.get('y', 0)
                            w = region.get('w', 0)
                            h = region.get('h', 0)
                            
                            if w > 0 and h > 0:
                                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
                                emotion = face.get('dominant_emotion', 'Unknown')
                                age = face.get('age', '?')
                                gender_data = face.get('gender', {})
                                if isinstance(gender_data, dict):
                                    gender = max(gender_data.items(), key=lambda item: item[1])[0] if gender_data else 'Unknown'
                                else:
                                    gender = gender_data
                                    
                                info_text = f"{gender}, {age} | {emotion}"
                                (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), (0, 255, 255), -1)
                                cv2.putText(image, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                
                    # Convert BGR to RGB for Streamlit display
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                    
                except ValueError:
                    st.warning("No face detected in the image.")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
