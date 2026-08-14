import av
import cv2
import streamlit as st
from utils.ui import (
    configure_page,
    render_sidebar_info,
    get_image_base64,
)
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from utils.config import DEMO_TRACKING
from utils.hand_tracking import HandTracker

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

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    import numpy as np

    class HandTrackingProcessor:
        def __init__(self):
            self.tracker = HandTracker()
            self.canvas = None
            self.xp, self.yp = 0, 0
            self.brush_thickness = 15
            self.draw_color = (255, 0, 255)
            self.typed_text = ""
            self.delay_counter = 0
            self.app_alert = ""
            self.alert_counter = 0
            
            # Setup keyboard keys
            self.keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                         ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
                         ["Z", "X", "C", "V", "B", "N", "M"]]
            self.button_list = []
            for i, row in enumerate(self.keys):
                for j, key in enumerate(row):
                    self.button_list.append({"x": 50 * j + 20, "y": 50 * i + 20, "text": key})

        def draw_keyboard(self, img):
            for button in self.button_list:
                x, y = button["x"], button["y"]
                cv2.rectangle(img, (x, y), (x + 40, y + 40), (255, 0, 255), cv2.FILLED)
                cv2.putText(img, button["text"], (x + 10, y + 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            return img

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # Mirror image
            if self.canvas is None:
                self.canvas = np.zeros_like(img)

            img, all_hands = self.tracker.find_hands(img, draw=True)
            
            # Decrease delay counters
            if self.delay_counter > 0:
                self.delay_counter -= 1
            if self.alert_counter > 0:
                self.alert_counter -= 1
            else:
                self.app_alert = ""

            left_hand = None
            right_hand = None
            for hand in all_hands:
                if hand["type"] == "Left":
                    left_hand = hand
                else:
                    right_hand = hand

            if left_hand and right_hand:
                # Two hands mode: Drawing
                fingers_left = self.tracker.fingers_up(left_hand)
                fingers_right = self.tracker.fingers_up(right_hand)
                
                # Left hand sets color
                if fingers_left[1] and not fingers_left[2]:
                    self.draw_color = (255, 0, 0) # Blue
                elif fingers_left[1] and fingers_left[2] and not fingers_left[3]:
                    self.draw_color = (0, 255, 0) # Green
                elif fingers_left[1] and fingers_left[2] and fingers_left[3] and not fingers_left[4]:
                    self.draw_color = (0, 0, 255) # Red
                elif fingers_left[1] and fingers_left[2] and fingers_left[3] and fingers_left[4]:
                    self.draw_color = (0, 0, 0) # Eraser
                    
                # Right hand draws
                if fingers_right[1] and not fingers_right[2]:
                    x1, y1 = right_hand["lmList"][8][0:2]
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1
                    if self.draw_color == (0, 0, 0):
                        cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, 50)
                    else:
                        cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)
                    self.xp, self.yp = x1, y1
                else:
                    self.xp, self.yp = 0, 0

            elif right_hand:
                # Right hand only: Keyboard
                self.xp, self.yp = 0, 0
                img = self.draw_keyboard(img)
                fingers = self.tracker.fingers_up(right_hand)
                
                # Erase with pinky only
                if fingers == [0, 0, 0, 0, 1] and self.delay_counter == 0:
                    self.typed_text = self.typed_text[:-1]
                    self.delay_counter = 10
                    
                # Index up for hover/typing
                if fingers[1]:
                    x8, y8 = right_hand["lmList"][8][0:2]
                    x12, y12 = right_hand["lmList"][12][0:2]
                    for button in self.button_list:
                        bx, by = button["x"], button["y"]
                        if bx < x8 < bx + 40 and by < y8 < by + 40:
                            cv2.rectangle(img, (bx, by), (bx + 40, by + 40), (175, 0, 175), cv2.FILLED)
                            cv2.putText(img, button["text"], (bx + 10, by + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            
                            # Click distance between index and middle
                            l = np.hypot(x12 - x8, y12 - y8)
                            if l < 30 and self.delay_counter == 0:
                                cv2.rectangle(img, (bx, by), (bx + 40, by + 40), (0, 255, 0), cv2.FILLED)
                                cv2.putText(img, button["text"], (bx + 10, by + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                                self.typed_text += button["text"]
                                self.delay_counter = 10

                # Display typed text
                cv2.rectangle(img, (20, 200), (600, 260), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, self.typed_text, (30, 245), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            elif left_hand:
                # Left hand only: Open apps
                self.xp, self.yp = 0, 0
                fingers = self.tracker.fingers_up(left_hand)
                if fingers == [0, 1, 0, 0, 0]:
                    self.app_alert = "Word Opened!"
                    self.alert_counter = 30
                elif fingers == [0, 1, 1, 0, 0]:
                    self.app_alert = "Excel Opened!"
                    self.alert_counter = 30
                elif fingers == [0, 1, 1, 1, 0]:
                    self.app_alert = "Firefox Opened!"
                    self.alert_counter = 30
                elif fingers == [0, 0, 0, 0, 0]:
                    self.app_alert = "Firefox Closed!"
                    self.alert_counter = 30

            else:
                self.xp, self.yp = 0, 0

            # Draw app alert
            if self.app_alert:
                cv2.putText(img, f"System Alert: {self.app_alert}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Merge canvas and image
            imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, self.canvas)

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
