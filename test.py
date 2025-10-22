# D:\FedXViT\test.py
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from models.hybrid_model import HybridModel
from sklearn.metrics import classification_report
import sys
import datetime

# --- 1. SETUP LOGGING ---
# Get a timestamp for the log file
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGFILE = f"test_log_{TIMESTAMP}.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect all print() statements to both terminal and log file
sys.stdout = Logger(LOGFILE)

print(f"--- [TEST RUN] ---")
print(f"Log file: {LOGFILE}\n")

# --- 2. SETUP DEVICE ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

def load_all_test_data(base_path):
    """Loads and combines the 'Testing' data from all clients."""
    print(f"📂 Loading all client test datasets from {base_path}...")
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the "Testing" dataset from all 3 client data folders
    client_0_test = ImageFolder(os.path.join(base_path, "CLIENT_0_DATA", "Testing"), transform=test_transforms)
    client_1_test = ImageFolder(os.path.join(base_path, "CLIENT_1_DATA", "Testing"), transform=test_transforms)
    client_2_test = ImageFolder(os.path.join(base_path, "CLIENT_2_DATA", "Testing"), transform=test_transforms)
    
    # Combine them into one single, large test set
    combined_test_dataset = ConcatDataset([client_0_test, client_1_test, client_2_test])
    
    print(f"✅ Combined test dataset created with {len(combined_test_dataset)} total images.")
    return DataLoader(combined_test_dataset, batch_size=16, shuffle=False)

if __name__ == "__main__":
    # 3. Load the model architecture
    model = HybridModel(num_classes=2)
    
    # 4. Load the final federated weights
    try:
        model.load_state_dict(torch.load("final_federated_model.pth", map_location=DEVICE))
        print("✅ Successfully loaded 'final_federated_model.pth'")
    except Exception as e:
        print(f"🚨 Error: Could not load 'final_federated_model.pth'. {e}")
        print("Please make sure the file was saved from your server.py run.")
        sys.stdout.log.close() # Close the log file
        exit()

    model.to(DEVICE)
    model.eval()

    # 5. Load all test data
    test_loader = load_all_test_data(r"D:\FedXViT\SplitData")

    # 6. Evaluate the model
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    print("--- 🚀 Starting Final Evaluation on all Test Data ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    class_names = ['Healthy', 'Tumor']
    
    print("\n" + "="*40)
    print("      FINAL FEDERATED MODEL PERFORMANCE")
    print("="*40)
    print(f"  Accuracy on Combined Test Set: {accuracy * 100:.2f}%")
    print(f"  Average Loss on Combined Test Set: {avg_loss:.4f}")
    print("="*40)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("--- [TEST RUN COMPLETE] ---")
    
    sys.stdout.log.close() # Close the log file