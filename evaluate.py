import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader

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

def evaluate_model(model_path='best_model.pth', batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Get Data (we only need test data here, but get_data returns all)
    _, _, X_test, y_test = get_data()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize Model and Load Weights
    input_dim = X_test.shape[2]
    num_classes = len(np.unique(y_test))
    
    model = HARTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Please train the model first.")
        return
        
    model.eval()
    
    # 3. Generate Predictions
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (Macro)")
    print(f"Recall:    {rec:.4f} (Macro)")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=ACTIVITY_NAMES))
    
    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_NAMES, yticklabels=ACTIVITY_NAMES)
    plt.title('Confusion Matrix - HAR Transformer')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to 'confusion_matrix.png'.")

if __name__ == '__main__':
    evaluate_model()
