import streamlit as st
from utils.ui import configure_page, render_sidebar_info, get_image_base64
from utils.config import DEMO_DROWSINESS

configure_page("CV Hub | Road Safety", "💤")
render_sidebar_info()

st.title("💤 Road Safety (Fatigue Detection)")
st.markdown(
    "Driver safety system tracking Eye Aspect Ratio (EAR) to prevent microsleep events."
)

st.markdown("---")

tab1, tab2 = st.tabs(["📖 Instructions & Demo", "🚀 Execution"])

with tab1:
    st.markdown("### How it works")
    st.markdown(
        "This module monitors driver fatigue by analyzing facial landmarks. It calculates the Eye Aspect Ratio (EAR) in real-time to detect prolonged eye closure, triggering an alert to prevent microsleep events."
    )
    st.markdown("#### Demonstration:")
    gif_b64 = get_image_base64(DEMO_DROWSINESS)
    st.markdown(
        f'<img src="{gif_b64}" width="100%" style="border-radius: 8px;">',
        unsafe_allow_html=True,
    )

with tab2:
    st.info(
        "Interactive execution mode for Fatigue Detection will be implemented here."
    )
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Driver Monitoring Feed")
        st.warning("Awaiting camera input...")

    with col2:
        st.markdown("### EAR Telemetry")
        st.info("Real-time Eye Aspect Ratio graph will be displayed here.")
        # Placeholder for EAR line chart

        st.markdown("### System Status")
        st.success("✅ Driver is alert")
