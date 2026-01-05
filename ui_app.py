# ui_app.py
import streamlit as st
import tempfile
from datetime import datetime

from inference import analyze_video
from ml_model import ml_predict
from config import ROAD_TYPES

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Traffic Signal AI Dashboard",
    layout="wide"
)

st.title("🚦 Traffic Signal AI Dashboard")
st.caption("Video-based traffic analysis and ML-driven signal recommendation")

# -------------------------------
# Sidebar (Inputs)
# -------------------------------
st.sidebar.header("Input Parameters")

road_type = st.sidebar.selectbox(
    "Road Type",
    ROAD_TYPES
)

video_file = st.sidebar.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

run_button = st.sidebar.button("Run Analysis")

# -------------------------------
# Main Panel
# -------------------------------
if run_button:

    if video_file is None:
        st.warning("Please upload a traffic video.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    with st.spinner("Processing video and running ML model..."):
        counts = analyze_video(video_path)
        suggestion = ml_predict(
            counts=counts,
            road_type=road_type
        )

    # -------------------------------
    # Layout: Metrics + Result
    # -------------------------------
    st.subheader("Detected Traffic Composition")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cars", counts["car"])
    col2.metric("Motorcycles", counts["motorcycle"])
    col3.metric("Buses", counts["bus"])
    col4.metric("Trucks", counts["truck"])

    st.divider()

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Context Information")
        st.write(f"**Road Type:** {road_type}")
        st.write(f"**Time of Analysis:** {datetime.now().strftime('%H:%M')}")

    with colB:
        st.subheader("Signal Recommendation")
        st.success(suggestion)

    st.caption(
        "This recommendation is generated using a supervised ML model trained "
        "to generalize traffic patterns from vehicle composition and context."
    )

else:
    st.info(
        "Upload a traffic video and select the road type from the sidebar "
        "to generate a signal timing recommendation."
    )
