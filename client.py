import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
import os
from models.hybrid_model import HybridModel 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Client running on device: {DEVICE}")

def load_client_data(data_path):
    """Loads a client's training and validation datasets."""
    print(f"📂 Client loading data from: {data_path}")
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_path, "Training")
    val_dir = os.path.join(data_path, "Validation")

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(DEVICE)
        self.model.train()

        print(f"--- Client training (1 epoch) on {len(self.train_loader.dataset)} images... ---")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5) 
        criterion = torch.nn.CrossEntropyLoss()
        
        last_loss = 0.0
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        
        print(f"--- Client training finished (Last loss: {last_loss:.4f}) ---")
        import mlflow
        mlflow.log_metric("local_loss", last_loss)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        print(f"--- Client evaluating on {len(self.val_loader.dataset)} validation images... ---")
        self.set_parameters(parameters)
        self.model.to(DEVICE)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        if total == 0:
            return 0.0, 0, {"accuracy": 0.0}

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"--- Client evaluation finished: Acc={accuracy:.4f}, Loss={avg_loss:.4f} ---")
        # Return all relevant metrics to the server
        return float(avg_loss), total, {"accuracy": float(accuracy), "loss": float(avg_loss)}