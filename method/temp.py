import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --- Scikit-learn Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Add any other sklearn models you plan to use by importing them here

# --- Configuration ---
FILE_PATH = "/model/lxj/LLM_comment_generate/temp/semantic_features/origin_qwen_code_semantic_cassandra.pt" # Replace with your actual file path

# Specify the machine learning methods to use (ensure they are imported above)
# Valid names are class names from scikit-learn
ML_METHODS_TO_USE = ["LogisticRegression", "RandomForestClassifier", "SVC"]

# Specify the deep learning methods to use (custom implemented or from a library)
DL_METHODS_TO_USE = ["SimpleNN"] # For now, we'll implement SimpleNN

# --- Device Configuration for PyTorch ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Seed for reproducibility ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Deep Learning Model Definition ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # Added dropout for regularization
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3) # Added dropout
        self.fc3 = nn.Linear(64, num_classes) # Output layer
        # For binary classification with BCEWithLogitsLoss, num_classes should be 1
        # For multi-class classification with CrossEntropyLoss, num_classes is the actual number of classes
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x) # Raw logits
        return x

# --- Helper Function to Load Data ---
def load_data(file_path):
    """Loads data from the .pt file."""
    try:
        data = torch.load(file_path, map_location=DEVICE) # Load directly to device if possible
        print(f"Successfully loaded data from {file_path}")
        return data['features'], data['labels']
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure 'semantic.pt' exists or create a dummy file using the provided script.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

# --- Helper Function for Training and Evaluating Scikit-learn Models ---
def train_evaluate_ml_model(model_name, X_train, y_train, X_test, y_test):
    """Trains and evaluates a scikit-learn model."""
    try:
        # Instantiate model
        if model_name == "LogisticRegression":
            model = LogisticRegression(random_state=SEED, max_iter=1000, solver='liblinear')
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=SEED, n_estimators=100)
        elif model_name == "SVC":
            model = SVC(random_state=SEED, probability=True) # probability=True for f1-score if needed
        elif model_name == "GaussianNB":
            model = GaussianNB()
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        else:
            raise ValueError(f"Unsupported ML model: {model_name}")

        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred, average='binary') # 'weighted' for multiclass, 'binary' for binary
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
        return accuracy, precision, recall, f1
    except Exception as e:
        print(f"Error training/evaluating {model_name}: {e}")
        return np.nan, np.nan, np.nan, np.nan


# --- Helper Function for Training and Evaluating Deep Learning Models ---
def train_evaluate_dl_model(model_instance, train_loader, test_loader, num_classes, epochs=20, lr=0.001):
    """Trains and evaluates a PyTorch deep learning model."""
    model_instance.to(DEVICE)
    # Determine loss function and target preparation based on num_classes
    if num_classes == 1: # Binary classification, assumes model outputs 1 logit
        criterion = nn.BCEWithLogitsLoss()
    else: # Multi-class classification, assumes model outputs raw logits for each class
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=lr)
    model_name = model_instance.__class__.__name__
    print(f"Training {model_name}...")

    for epoch in range(epochs):
        model_instance.train()
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model_instance(batch_features)

            # Adjust labels for loss function
            if num_classes == 1: # Binary
                # BCEWithLogitsLoss expects target to be float and same shape as output
                loss = criterion(outputs, batch_labels.float().unsqueeze(1))
            else: # Multi-class
                # CrossEntropyLoss expects target to be long and of shape (N)
                loss = criterion(outputs, batch_labels.long())
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Basic epoch logging
        if (epoch + 1) % (epochs // 5 if epochs >=5 else 1) == 0 or epoch == epochs -1 :
             print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}")

    # Evaluation
    model_instance.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(DEVICE)
            outputs = model_instance(batch_features)
            
            if num_classes == 1: # Binary
                preds = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()
            else: # Multi-class
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Ensure all_preds is not empty before calculating metrics
    if len(all_preds) == 0:
        print("Warning: No predictions made during DL model evaluation. Check test_loader and model output.")
        accuracy = np.nan
        f1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        # accuracy = accuracy_score(all_labels, all_preds)
        # f1 = f1_score(all_labels, all_preds, average='binary') # 'weighted' for multiclass
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return accuracy, precision, recall, f1

# --- Main ---
if __name__ == "__main__":
    # 1. Load data
    all_features_dict, labels_tensor = load_data(FILE_PATH)
    
    # Convert labels tensor to NumPy array for scikit-learn
    # For PyTorch models, we'll use the tensor directly
    labels_np = labels_tensor.cpu().numpy()
    
    # Determine number of classes
    unique_labels = torch.unique(labels_tensor)
    num_classes_actual = len(unique_labels)
    print(f"Detected {num_classes_actual} classes: {unique_labels.cpu().numpy()}")

    # For binary classification handled by SimpleNN with BCEWithLogitsLoss, output dim is 1
    # For multi-class with CrossEntropyLoss, output dim is num_classes_actual
    nn_output_dim = 1 if num_classes_actual == 2 else num_classes_actual


    results = []
    feature_categories = list(all_features_dict.keys()) # ['shallow', 'middle', 'deep', 'last_token']

    # 4. Perform prediction task for each feature category
    for category_name in feature_categories:
        print(f"\n--- Processing Feature Category: {category_name.upper()} ---")
        
        # 2. Extract specific features for the current category
        # Convert to NumPy array for scikit-learn
        features_np = all_features_dict[category_name].cpu().numpy()
        
        # Ensure features are 2D
        if features_np.ndim == 1:
            features_np = features_np.reshape(-1, 1)
        if features_np.shape[0] != len(labels_np):
            print(f"Warning: Mismatch in number of samples for {category_name} ({features_np.shape[0]}) and labels ({len(labels_np)}). Skipping.")
            continue
        if features_np.shape[0] == 0:
            print(f"Warning: No samples found for {category_name}. Skipping.")
            continue

        # Split data for this category
        X_train, X_test, y_train, y_test = train_test_split(
            features_np, labels_np, test_size=0.2, random_state=SEED, stratify=labels_np
        )
        
        # Scale features for ML models (especially important for SVM, Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- 5. Implement Machine Learning methods ---
        print(f"\n-- Machine Learning Models for {category_name} --")
        for ml_model_name in ML_METHODS_TO_USE:
            accuracy, precision, recall, f1 = train_evaluate_ml_model(
                ml_model_name, X_train_scaled, y_train, X_test_scaled, y_test
            )
            results.append({
                "Feature Category": category_name,
                "Model Type": "Machine Learning",
                "Model Name": ml_model_name,
                "Accuracy":accuracy,
                "F1-score": f1,
                "Recall": recall,
                "precision": precision
            })

        # --- 6. Implement Deep Learning methods ---
        print(f"\n-- Deep Learning Models for {category_name} --")
        
        # Prepare data for PyTorch (use non-scaled features directly, as NNs can learn scaling)
        # Convert back to tensors and move to DEVICE
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train) # Type will be adjusted in training loop
        y_test_tensor = torch.tensor(y_test)   # Type will be adjusted in training loop

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Handle cases with too few samples for batching
        batch_size = 32
        if len(train_dataset) < batch_size:
             print(f"Warning: Training dataset size ({len(train_dataset)}) is smaller than batch_size ({batch_size}). Adjusting batch_size.")
             batch_size_train = max(1, len(train_dataset)) # Ensure batch size is at least 1
        else:
             batch_size_train = batch_size
        
        if len(test_dataset) < batch_size:
             print(f"Warning: Test dataset size ({len(test_dataset)}) is smaller than batch_size ({batch_size}). Adjusting batch_size.")
             batch_size_test = max(1, len(test_dataset))
        else:
             batch_size_test = batch_size

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

        input_dim = X_train_tensor.shape[1]


        for dl_model_name in DL_METHODS_TO_USE:
            if dl_model_name == "SimpleNN":
                # Pass nn_output_dim (1 for binary, num_classes_actual for multi-class)
                model = SimpleNN(input_dim=input_dim, num_classes=nn_output_dim)
            else:
                print(f"Warning: Deep learning model {dl_model_name} not implemented. Skipping.")
                continue
            
            accuracy, precision, recall, f1 = train_evaluate_dl_model(
                model, train_loader, test_loader, num_classes=nn_output_dim, epochs=20 # Can increase epochs
            )
            results.append({
                "Feature Category": category_name,
                "Model Type": "Deep Learning",
                "Model Name": dl_model_name,
                "Accuracy": accuracy,
                "F1-score": f1,
                "Recall": recall,
                "precision": precision
            })
            del model # Free up GPU memory if applicable

    # 7. Compare performance (using a Pandas DataFrame for nice printing)
    print("\n\n--- Overall Performance Comparison ---")
    results_df = pd.DataFrame(results)
    
    # Sort for better readability
    results_df = results_df.sort_values(by=["Feature Category", "Model Type", "F1-score"], ascending=[True, True, False])
    
    print(results_df.to_string()) # .to_string() prints the whole DataFrame

    # You can also save this to a CSV
    # results_df.to_csv("model_performance_comparison.csv", index=False)
    # print("\nResults saved to model_performance_comparison.csv")