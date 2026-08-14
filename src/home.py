import streamlit as st

from utils.ui import configure_page, render_sidebar_info, cctv_card
from utils.config import (
    IMG_PREVIEW_CLASSIFICATION,
    IMG_PREVIEW_TRAFFIC,
    IMG_PREVIEW_TRACKING,
    IMG_PREVIEW_DROWSINESS,
)

# Configure Page
configure_page("CV Hub | Home", "👁️‍🗨️")
render_sidebar_info()

# Header
st.title("👁️‍🗨️ Computer Vision Command Center")
st.markdown(
    """
Monitor and control real-time analytics across multiple computer vision modules.
Select a specific feed from the sidebar to interact with the models directly.
"""
)

st.markdown("---")

with st.spinner("⏳ Estabelecendo conexão com as câmeras de segurança..."):
    col1, col2 = st.columns(2)

    with col1:
        cctv_card(
            title="📍 CAM 01: Object Classification",
            image_path=IMG_PREVIEW_CLASSIFICATION,
        )

        cctv_card(
            title="📍 CAM 03: Human-Machine Interaction",
            image_path=IMG_PREVIEW_TRACKING,
        )

    with col2:
        cctv_card(
            title="📍 CAM 02: Traffic Analysis",
            image_path=IMG_PREVIEW_TRAFFIC,
        )

        cctv_card(
            title="📍 CAM 04: Biometric Monitoring",
            image_path=IMG_PREVIEW_DROWSINESS,
        )
