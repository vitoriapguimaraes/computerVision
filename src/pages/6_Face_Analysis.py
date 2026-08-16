import streamlit as st
import cv2
import av
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from deepface import DeepFace

st.set_page_config(page_title="Face Analysis", page_icon="🧑", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 0;
}
.sub-header {
    font-size: 1.1rem;
    color: #6B7280;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Real-time Face Analysis 🧑</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect emotions, age, and gender using DeepFace</p>', unsafe_allow_html=True)

st.markdown("""
> **Note:** The first time you activate the camera, DeepFace will download the required pre-trained weights.
> To keep the video feed smooth, the analysis is executed on a throttled schedule (not every frame).
""")

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
                results = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], enforce_detection=False, silent=True)
                
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

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="face-analysis",
        mode=1, # SENDRECV
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
