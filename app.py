import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from data_processing import get_data
from model import HARTransformer

# Page Config
st.set_page_config(page_title="HAR Transformer Dashboard", layout="wide")

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

SIGNALS = [
    "Body Acc X", "Body Acc Y", "Body Acc Z",
    "Body Gyro X", "Body Gyro Y", "Body Gyro Z",
    "Total Acc X", "Total Acc Y", "Total Acc Z"
]

@st.cache_resource
def load_har_model(model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_test, y_test = get_data()
    
    input_dim = X_test.shape[2]
    num_classes = len(np.unique(y_test))
    model = HARTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, X_test, y_test, device
    except FileNotFoundError:
        return None, None, None, None

def main():
    st.title("🏃 Human Activity Recognition - Transformer Analysis")
    st.markdown("""
    This dashboard allows you to analyze how the **Multimodal Transformer** predicts human activities based on smartphone sensor data.
    """)

    model, X_test, y_test, device = load_har_model()

    if model is None:
        st.error("Model file 'best_model.pth' not found. Please train the model first.")
        return

    # Sidebar for controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Pick Random Sample"):
        st.session_state.sample_idx = random.randint(0, len(X_test) - 1)
    
    if 'sample_idx' not in st.session_state:
        st.session_state.sample_idx = 0

    sample_idx = st.sidebar.number_input("Sample Index", 0, len(X_test)-1, st.session_state.sample_idx)
    st.session_state.sample_idx = sample_idx

    # Get data
    sample = X_test[sample_idx]
    label = y_test[sample_idx]

    # Predict
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(sample_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)

    # Layout: Top Row - Result and Probabilities
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction Result")
        is_correct = (pred_idx == label)
        
        if is_correct:
            st.success(f"**Predicted:** {ACTIVITY_NAMES[pred_idx]}")
        else:
            st.error(f"**Predicted:** {ACTIVITY_NAMES[pred_idx]}")
        
        st.info(f"**Ground Truth:** {ACTIVITY_NAMES[label]}")
        st.metric("Confidence", f"{probs[pred_idx]*100:.2f}%")

    with col2:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            'Activity': ACTIVITY_NAMES,
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index('Activity'))

    # Layout: Bottom Row - Sensor Data Visualization
    st.divider()
    st.subheader("Sensor Signal Visualization (2.56s Window)")
    
    # Create tabs for Accel and Gyro
    tab1, tab2, tab3 = st.tabs(["Accelerometer (Body)", "Gyroscope", "Accelerometer (Total)"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(3):
            ax.plot(sample[:, i], label=SIGNALS[i])
        ax.set_title("Body Accelerometer Signals")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(3, 6):
            ax.plot(sample[:, i], label=SIGNALS[i])
        ax.set_title("Body Gyroscope Signals")
        ax.legend()
        st.pyplot(fig)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(6, 9):
            ax.plot(sample[:, i], label=SIGNALS[i])
        ax.set_title("Total Accelerometer Signals")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
