import streamlit as st
from datetime import datetime
import base64
import os


def configure_page(page_title="Computer Vision Hub", page_icon="👁️"):
    """Configures the standard page settings for all pages."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a professional, "CCTV/Dashboard" dark aesthetic
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #1E3A8A;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3B82F6;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        /* Custom styling for metrics */
        div[data-testid="stMetricValue"] {
            color: #10B981; /* Emerald green for active metrics */
            font-size: 2.5rem !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #9CA3AF;
            font-size: 1.1rem;
        }
        /* CCTV Frame Style */
        .cctv-frame {
            border: 2px solid #374151;
            border-radius: 8px;
            padding: 10px;
            background-color: #111827;
            position: relative;
        }
        .cctv-label {
            position: absolute;
            top: 15px;
            left: 15px;
            background-color: rgba(220, 38, 38, 0.8); /* Red indicator */
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            z-index: 10;
        }
        .cctv-timestamp {
            position: absolute;
            bottom: 15px;
            right: 15px;
            color: rgba(255, 255, 255, 0.7);
            font-family: monospace;
            font-size: 0.9rem;
            z-index: 10;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar_info():
    """Renders standard information in the sidebar."""
    st.sidebar.caption("Trabalho de github.com/vitoriapguimaraes")
    st.sidebar.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vitoriapguimaraes) "
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vitoriapguimaraes/)"
    )


@st.cache_data(show_spinner=False)
def get_image_base64(image_path):
    """Loads a local image and converts it to base64 for HTML embedding."""
    if image_path.startswith("http"):
        return image_path

    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return "https://via.placeholder.com/600x250.png?text=FEED+LOST"

    with open(abs_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        ext = abs_path.split(".")[-1]
        return f"data:image/{ext};base64,{encoded}"


def cctv_card(title, image_path, link_text="View Feed"):
    """Renders a CCTV styled card."""

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    b64_src = get_image_base64(image_path)

    st.markdown(f"### {title}")

    # CCTV Container
    html = f"""
    <div class="cctv-frame" style="margin-bottom: 20px;">
        <div class="cctv-label">● REC</div>
        <img src="{b64_src}" style="width: 100%; height: 250px; object-fit: cover; border-radius: 4px;" alt="{title}">
        <div class="cctv-timestamp">{current_time}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
