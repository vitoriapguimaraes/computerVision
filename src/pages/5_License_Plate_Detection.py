import os
import cv2
import av
import streamlit as st
import pytesseract
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from utils.ui import configure_page, render_sidebar_info

configure_page("License Plate Detection", "🚗")
render_sidebar_info()

st.title("🚗 License Plate Text Detection (OCR)")
st.markdown("Extraction of characters from license plates using OpenCV contour detection and Tesseract OCR.")

# Try to optimize Tesseract for short alphanumeric text
CUSTOM_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

import re

def process_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200) 
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    location = None
    for contour in contours:
        # First try strict polygon approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 1.5 < aspect_ratio < 5.5 and w > 50 and h > 15:
                location = approx
                break
                
    # Fallback to bounding box proportions if polygon approximation fails
    if location is None:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 2.5 < aspect_ratio < 4.5 and w > 80 and h > 20:
                # Create a fake 4-point contour from the bounding box
                location = np.array([[[x, y]], [[x, y+h]], [[x+w, y+h]], [[x+w, y]]])
                break
                
    extracted_text = ""
    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        
        (x, y) = np.where(mask == 255)
        if len(x) > 0 and len(y) > 0:
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = gray[topx:bottomx+1, topy:bottomy+1]
            
            if cropped.size > 0:
                # Upscale image to improve Tesseract accuracy
                cropped = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                
                # Define preprocessing strategies (like the Jupyter notebook)
                strategies = [
                    # 1. Simple Otsu Thresholding
                    lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                    # 2. Blur + Otsu (Reduces noise)
                    lambda img: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                    # 3. Morphological BlackHat (Isolates dark text on light background)
                    lambda img: cv2.threshold(cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                    # 4. Morphological TopHat (Isolates light text on dark background)
                    lambda img: cv2.threshold(cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                    # 5. Erosion on Otsu (Thickens dark text characters)
                    lambda img: cv2.erode(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1),
                    # 6. Grayscale Raw
                    lambda img: img
                ]
                
                best_raw_text = ""
                
                for strategy in strategies:
                    processed_crop = strategy(cropped.copy())
                    
                    text = pytesseract.image_to_string(processed_crop, lang='por', config=CUSTOM_CONFIG)
                    text = text.upper().strip()
                    
                    # Use strict regex to find Brazilian Mercosul or old standard format
                    mercosul_match = re.search(r'[A-Z]{3}\d[A-Z]\d{2}', text)
                    old_match = re.search(r'[A-Z]{3}-?\d{4}', text)
                    
                    if mercosul_match:
                        extracted_text = mercosul_match.group(0)
                        break # Found perfect match, stop trying strategies!
                    elif old_match:
                        extracted_text = old_match.group(0)
                        break # Found perfect match, stop trying strategies!
                    else:
                        # Clean up non-alphanumeric if it fails to match exactly
                        clean_text = re.sub(r'[^A-Z0-9]', '', text)
                        if len(clean_text) > len(best_raw_text):
                            best_raw_text = clean_text # Save the best attempt just in case
                
                # If all strategies failed to produce a valid regex plate, fallback to the longest string found
                if not extracted_text:
                    extracted_text = best_raw_text
                
                cv2.rectangle(img, (topy, topx), (bottomy, bottomx), (0, 255, 0), 3)
                
                if extracted_text:
                    cv2.putText(img, extracted_text, (topy, topx - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return img, extracted_text

tab1, tab2 = st.tabs(["Instructions & Demo", "Execution"])

with tab1:
    how_it_works = """
This module performs the following pipeline on each image or frame:
1. **Grayscale & Blur:** Prepares the image for edge detection.
2. **Canny Edge Detection:** Highlights the edges of objects.
3. **Contour Finding:** Searches for rectangles or valid plate proportions.
4. **Multi-Strategy OCR:** Applies a fallback loop of computer vision filters until Tesseract extracts a valid Mercosul plate.
    """
    
    def render_mermaid_graph():
        st.markdown("#### OCR Fallback Pipeline (Decision Graph)")
        
        import streamlit.components.v1 as components
        
        mermaid_code = """
        graph LR
            A[Crop License Plate] --> B[Upscale 3x]
            B --> C[Select Next Strategy]
            
            subgraph Fallback Loop
                C --> D[Apply Filter]
                D --> E[Tesseract OCR]
                E --> F{Matches Regex?}
            end
            
            F -- Yes --> G((Success!))
            F -- No --> H{More Strategies?}
            H -- Yes -->|Next| C
            H -- No --> I((Fallback: Longest String))
            
            subgraph Available Strategies
                S1[1. Otsu Threshold]
                S2[2. Blur + Otsu]
                S3[3. Morph BlackHat]
                S4[4. Morph TopHat]
                S5[5. Erosion + Otsu]
                S6[6. Grayscale Raw]
            end
        """
        
        components.html(
            f"""
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'base', themeVariables: {{ primaryColor: '#2C3E50', primaryTextColor: '#fff', primaryBorderColor: '#7C3AED', lineColor: '#3B82F6', secondaryColor: '#10B981', tertiaryColor: '#fff' }} }});
            </script>
            <div class="mermaid" style="display: flex; justify-content: center;">
                {mermaid_code}
            </div>
            """,
            height=550,
            scrolling=True
        )

    render_instructions_tab(how_it_works, DEMO_LICENSE_PLATE, render_mermaid_graph)

with tab2:
    st.markdown("### Input Feed")
    
    input_method = st.radio(
        "Select Input Method:", ["Example Image", "Upload Image", "Webcam (WebRTC)"], horizontal=True
    )
    
    if input_method == "Webcam (WebRTC)":
        st.info("The video is streamed directly from your browser to the Docker container via WebRTC.")
        
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        class LicensePlateProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1) # Mirror
                
                processed_img, text = process_license_plate(img)
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        webrtc_streamer(
            key="license_plate_detection",
            video_processor_factory=LicensePlateProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
    else:
        st.markdown("### Analysis Results")
        col1, col2 = st.columns([1, 1])
        
        img_array = None
        
        if input_method == "Example Image":
            examples_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples_ocr_reading")
            if os.path.exists(examples_dir):
                examples = [f for f in os.listdir(examples_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                selected_example = st.selectbox("Choose an example image:", examples)
                
                if selected_example:
                    example_path = os.path.join(examples_dir, selected_example)
                    pil_img = Image.open(example_path)
                    img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                st.error(f"Directory not found: {examples_dir}")
                
        elif input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
        if img_array is not None:
            with col1:
                st.markdown("#### Input Image")
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            with col2:
                with st.spinner("Processing image..."):
                    processed_img, extracted_text = process_license_plate(img_array.copy())
                    
                    st.markdown("#### Processed Image")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    if extracted_text:
                        st.success(f"**Extracted Text:** {extracted_text}")
                    else:
                        st.warning("No license plate text detected.")
