import torch
import numpy as np
import random
from data_processing import get_data
from model import HARTransformer

ACTIVITY_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

def run_inference(model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print("Loading test data for inference...")
    _, _, X_test, y_test = get_data()
    
    # 2. Initialize and Load Model
    input_dim = X_test.shape[2]
    num_classes = len(np.unique(y_test))
    model = HARTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}\n")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        return

    # 3. Interactive Testing
    print("--- HAR Transformer Interactive Test ---")
    print("Press Enter to test a random sample, or type 'q' to quit.")
    
    while True:
        user_input = input("\n[Test Random Sample] > ")
        if user_input.lower() == 'q':
            break
            
        # Pick a random sample from the test set
        idx = random.randint(0, len(X_test) - 1)
        sample = X_test[idx]
        label = y_test[idx]
        
        # Prepare sample for model (add batch dimension)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(sample_tensor)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[0]
        
        pred_idx = predicted.item()
        confidence = probs[pred_idx].item() * 100
        
        print(f"Sample Index: {idx}")
        print(f"Ground Truth: {ACTIVITY_NAMES[label]}")
        print(f"Prediction:   {ACTIVITY_NAMES[pred_idx]} ({confidence:.2f}% confidence)")
        
        if pred_idx == label:
            print("✅ Correct!")
        else:
            print("❌ Incorrect.")

if __name__ == '__main__':
    run_inference()
