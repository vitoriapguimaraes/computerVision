import os
import cv2
import av
import streamlit as st
import streamlit.components.v1 as components
import pytesseract
import numpy as np
import re
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from skimage.segmentation import clear_border

from utils.ui import (configure_page, render_sidebar_info, render_instructions_tab)
from settings.config import IMG_PREVIEW_LICENSE_PLATE_READING

configure_page("License Plate Detection", "🚗")
render_sidebar_info()

st.title("License Plate Text Detection (OCR)")
st.markdown("Extraction of characters from license plates using OpenCV contour detection and Tesseract OCR.")

# Try to optimize Tesseract for short alphanumeric text
CUSTOM_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

def apply_plate_heuristics(text: str) -> str:
    # Strip all non-alphanumeric characters
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    raw_candidates = []
    # If longer than 7, try every 7-character substring
    if len(clean_text) >= 7:
        for start in range(len(clean_text) - 6):
            raw_candidates.append(clean_text[start:start+7])
            
    # Also try dropping one character if we have exactly 8 (handles inserted noise like logos)
    if len(clean_text) == 8:
        for i in range(8):
            raw_candidates.append(clean_text[:i] + clean_text[i+1:])
            
    candidates = []
    dict_char_to_int = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

    for cand in raw_candidates:
        plate = list(cand)
        # Position 0, 1, 2 must be LETTERS
        for i in range(3):
            if plate[i] in dict_int_to_char:
                plate[i] = dict_int_to_char[plate[i]]
                
        # Position 3 must be NUMBER
        if plate[3] in dict_char_to_int:
            plate[3] = dict_char_to_int[plate[3]]
            
        # Position 4 can be LETTER (Mercosul) or NUMBER (Old).
        
        # Position 5, 6 must be NUMBERS
        for i in range(5, 7):
            if plate[i] in dict_char_to_int:
                plate[i] = dict_char_to_int[plate[i]]
                
        candidates.append("".join(plate))
            
    # Check if any candidates perfectly match Mercosul first (higher priority)
    for cand in candidates:
        if re.search(r'^[A-Z]{3}\d[A-Z]\d{2}$', cand):
            return cand
            
    # Then check old Brazilian format
    for cand in candidates:
        if re.search(r'^[A-Z]{3}\d{4}$', cand):
            return cand
            
    # If no perfect match, return the first candidate (or the clean_text)
    if candidates:
        return candidates[0]
    return clean_text

def find_license_plate_contour_morphology(gray):
    # This pipeline is based on the Jupyter notebook experiments (BlackHat + Sobel)
    kernel_rectangular = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 13))
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_rectangular)
    
    sobel_x = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobel_x = np.absolute(sobel_x).astype('uint8')
    
    sobel_x = cv2.GaussianBlur(sobel_x, (5, 5), 0)
    sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, kernel_rectangular)
    
    _, threshold_otsu = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(threshold_otsu, kernel_square, iterations=2)
    dilation = cv2.dilate(erosion, kernel_square, iterations=2)
    
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_square)
    _, mask = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    threshold_bitwise = cv2.bitwise_and(dilation, dilation, mask=mask)
    dilation = cv2.dilate(threshold_bitwise, kernel_square, iterations=2)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    
    thresholding = clear_border(erosion)
    
    keypoints = cv2.findContours(thresholding, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if 2.5 <= aspect_ratio <= 4.0 and w > 50 and h > 15:
            return np.array([[[x, y]], [[x, y+h]], [[x+w, y+h]], [[x+w, y]]])
    return None

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
                
    # Second fallback: Morphological pipeline (BlackHat + Sobel + clear_border)
    if location is None:
        location = find_license_plate_contour_morphology(gray)

    # Third fallback: naive bounding box proportions if polygon approximation fails
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
                # Apply a tighter crop to remove top "BRASIL" text and side screws
                h_c, w_c = cropped.shape
                cropped = cropped[int(h_c*0.15):int(h_c*0.95), int(w_c*0.05):int(w_c*0.95)]
                
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
                    # Apply heuristics to clean text and fix common character confusions
                    text = apply_plate_heuristics(text)
                    
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
                        # Fallback to the heuristic cleaned text
                        if len(text) > len(best_raw_text):
                            best_raw_text = text # Save the best attempt just in case
                
                # If all strategies failed to produce a valid regex plate, fallback to the longest string found
                if not extracted_text:
                    # Enforce the 7 char limit on fallback if it's longer to avoid huge messy strings
                    extracted_text = best_raw_text[:7] if len(best_raw_text) >= 7 else best_raw_text
                
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
                
        mermaid_code = """
        graph LR
            A["Crop License Plate"] --> B["Upscale 3x"]
            B --> C["Select Next Strategy"]
            
            subgraph Fallback_Loop ["Fallback Loop"]
                C --> D["Apply Filter"]
                D --> E["Tesseract OCR"]
                E --> F{"Matches Regex?"}
            end
            
            F -->|"Yes"| G["🎉 Success!"]
            F -->|"No"| H{"More Strategies?"}
            H -->|"Yes (Next)"| C
            H -->|"No"| I["⚠️ Fallback: Longest String"]
            
            classDef default fill:#1E293B,stroke:#3B82F6,stroke-width:2px,color:#F8FAFC,rx:5,ry:5;
            classDef success fill:#059669,stroke:#10B981,stroke-width:2px,color:#fff;
            classDef warning fill:#B45309,stroke:#F59E0B,stroke-width:2px,color:#fff;
            classDef decision fill:#4C1D95,stroke:#8B5CF6,stroke-width:2px,color:#fff;
            
            class G success;
            class I warning;
            class F,H decision;
        """
        
        components.html(
            f"""
            <style>
                body {{
                    background-color: transparent; 
                    color: white; 
                    margin: 0; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
            </style>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ 
                    startOnLoad: true, 
                    theme: 'dark',
                    themeVariables: {{
                        lineColor: '#94A3B8'
                    }}
                }});
            </script>
            <div class="mermaid" style="display: flex; justify-content: center;">
                {mermaid_code}
            </div>
            """,
            height=650,
            scrolling=True
        )
        
        st.markdown(
            "**Available Strategies:** "
            "`1. Otsu Threshold` ➔ `2. Blur + Otsu` ➔ `3. Morph BlackHat` ➔ `4. Morph TopHat` ➔ `5. Erosion + Otsu` ➔ `6. Grayscale`"
        )

    render_instructions_tab(how_it_works, IMG_PREVIEW_LICENSE_PLATE_READING, render_mermaid_graph)

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
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_column_width=True)
                
            with col2:
                with st.spinner("Processing image..."):
                    processed_img, extracted_text = process_license_plate(img_array.copy())
                    
                    st.markdown("#### Processed Image")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    if extracted_text:
                        st.success(f"**Extracted Text:** {extracted_text}")
                    else:
                        st.warning("No license plate text detected.")
