import streamlit as st
import tempfile
from PIL import Image
import os

# Import your functions
from shark_functions import (
    process_video,
    add_point,
    close_polygon,
    clear_points,
    run_sam2,
    create_comparison_video,
    create_overlay_video,
    check_missing_frames,
    export_annotations,
    first_frame_array,
    polygon_points
)

st.set_page_config(page_title="Shark Annotation Tool", layout="wide")

st.title("Shark Video Annotation Tool 🦈")
st.write("Draw a polygon on ONE frame — get ALL frames annotated!")

# -----------------------------
# STEP 1: UPLOAD VIDEO
# -----------------------------
st.header("Step 1: Upload Video")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
fps = st.slider("FPS for frame extraction", 1, 10, 5)

if uploaded_video:
    st.video(uploaded_video)

if st.button("Process Video"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    img, status = process_video(video_path, fps)
    st.session_state["current_image"] = img
    st.success(status)

# -----------------------------
# STEP 2: DRAW POLYGON
# -----------------------------
st.header("Step 2: Draw Polygon")

if "current_image" in st.session_state:
    st.image(st.session_state["current_image"], caption="Current Frame")

    st.write("Streamlit cannot detect clicks on images yet, so enter coordinates manually.")

    x = st.number_input("X coordinate", min_value=0)
    y = st.number_input("Y coordinate", min_value=0)

    if st.button("Add Point"):
        fake_event = type("obj", (object,), {"index": (x, y)})
        img, msg = add_point(st.session_state["current_image"], fake_event)
        st.session_state["current_image"] = img
        st.image(img)
        st.write(msg)

    if st.button("Close Polygon"):
        img, msg = close_polygon()
        st.session_state["current_image"] = img
        st.image(img)
        st.write(msg)

    if st.button("Clear Points"):
        img, msg = clear_points()
        st.session_state["current_image"] = img
        st.image(img)
        st.write(msg)

# -----------------------------
# STEP 3: RUN SAM2
# -----------------------------
st.header("Step 3: Run SAM 2 Tracking")

if st.button("Run SAM 2"):
    msg, preview = run_sam2()
    st.write(msg)
    if preview:
        st.image(preview, caption="Preview Frames")

# -----------------------------
# STEP 4: CREATE VIDEOS
# -----------------------------
st.header("Step 4: Create Comparison Videos")

if st.button("Create Side-by-Side Comparison"):
    video_path, msg = create_comparison_video()
    st.write(msg)
    if video_path:
        st.video(video_path)

if st.button("Create Overlay Video"):
    video_path, msg = create_overlay_video()
    st.write(msg)
    if video_path:
        st.video(video_path)

# -----------------------------
# STEP 5: QUALITY CHECK
# -----------------------------
st.header("Step 5: Check Tracking Quality")

if st.button("Check Missing Frames"):
    report = check_missing_frames()
    st.text(report)

# -----------------------------
# STEP 6: EXPORT
# -----------------------------
st.header("Step 6: Export Annotations")

class_name = st.text_input("Class name", "shark")

if st.button("Export for Roboflow"):
    zip_path, msg = export_annotations(class_name)
    st.write(msg)
    if zip_path:
        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", f, file_name="annotations.zip")
