import streamlit as st
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from vision_classifier import VisionHARClassifier

# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = 'pose_landmarker.task'

# Standard Pose Connections
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), 
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32), (27, 29), (28, 30), (29, 31), (30, 32)
]

# Initialize Classifier
har_classifier = VisionHARClassifier()

# Page Config
st.set_page_config(page_title="NEURAL HAR - LIVE HUD", layout="wide", initial_sidebar_state="expanded")

# Custom Futuristic CSS
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle, #1a1c24 0%, #0e1117 100%); }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 0 0 10px #00f2ff; font-family: 'Courier New', Courier, monospace; }
    .stSidebar { background-color: rgba(0, 242, 255, 0.05); border-right: 1px solid #00f2ff; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def create_landmarker():
    # Auto-download model for deployment if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Neural Engine (Pose Landmarker)..."):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            import requests
            r = requests.get(url)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"Failed to load MediaPipe model: {e}")
        return None

def draw_pose_hud(image, landmarks):
    h, w, _ = image.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (255, 242, 0), -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(image, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), (230, 66, 245), 2)

def get_radar_html(probs):
    categories = list(probs.keys())
    values = list(probs.values())
    
    fig = go.Figure()

    # Outer Glow Layer
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.1)',
        line=dict(color='rgba(0, 212, 255, 0.5)', width=4),
        hoverinfo='skip'
    ))

    # Inner Solid Layer
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.3)',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=8, color='#00d4ff', symbol='diamond')
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="#333", 
                tickfont=dict(color="#00d4ff", size=10),
                angle=90, tickangle=90
            ),
            angularaxis=dict(
                gridcolor="#333", linecolor="#00d4ff",
                tickfont=dict(color="white", size=12, family="Courier New")
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=30, b=40),
        height=320,
        annotations=[
            dict(text="NEURAL_SENSE_V1", x=0.5, y=-0.15, showarrow=False, font=dict(color="#00d4ff", size=10))
        ]
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

def get_confusion_heatmap():
    z = [[496, 0, 0, 0, 0, 0], [0, 471, 10, 0, 0, 0], [0, 15, 517, 0, 0, 0], [0, 0, 0, 482, 5, 4], [0, 0, 0, 2, 490, 4], [0, 0, 0, 1, 0, 532]]
    classes = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    fig = px.imshow(z, x=classes, y=classes, color_continuous_scale='Blues', text_auto=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=350, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def main():
    st.title("⚡ NEURAL HAR: REAL-TIME HUD")
    
    col_vid, col_metrics = st.columns([1.5, 1])

    with col_vid:
        st.subheader("SYSTEM_INPUT: LIVE_FEED")
        video_placeholder = st.empty()
        
    with col_metrics:
        st.subheader("ANALYSIS_HUD")
        prediction_text = st.empty()
        st.write("---")
        st.markdown("**CONFIDENCE_RADAR**")
        radar_html_placeholder = st.empty()
        st.write("---")
        st.markdown("**MODEL_METRICS**")
        st.plotly_chart(get_confusion_heatmap(), use_container_width=True, key="static_heatmap")

    with st.sidebar:
        st.header("HUD_CONTROL")
        cam_index = st.number_input("Camera Index", value=0, step=1)
        use_webcam = st.checkbox("ENABLE_SYSTEM", value=True)
        show_skeleton = st.checkbox("SKELETON_OVERLAY", value=True)
        st.write("---")
        stop_button = st.button("TERMINATE_SYSTEM")

    if use_webcam:
        landmarker = create_landmarker()
        if landmarker is None: return

        cap = cv2.VideoCapture(int(cam_index))
        
        if not cap.isOpened():
            st.error(f"FATAL: Could not access camera at index {cam_index}. Check permissions or select a different index.")
            return

        frame_count = 0
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("FEED_INTERRUPTED: Check camera connection.")
                break
            
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            action = "IDLE"
            probs = {c: 0.1 for c in har_classifier.classes}
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                action, conf, probs = har_classifier.classify(landmarks)
                if show_skeleton: draw_pose_hud(frame, landmarks)

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            prediction_text.markdown(f"<h2 style='text-align:center;'>STATE: {action.upper()}</h2>", unsafe_allow_html=True)
            
            if frame_count % 5 == 0:
                with radar_html_placeholder:
                    components.html(get_radar_html(probs), height=320)

        cap.release()
        st.info("SYSTEM_OFFLINE: Camera released.")

if __name__ == "__main__":
    main()
