'''
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
    val_dir = os.path.join(data_path, "Validation") # We now use the Validation folder

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # --- THIS IS THE FIX ---
    # Return both loaders as a tuple
    return train_loader, val_loader
    # --- END OF FIX ---


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # New validation loader

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
            print("--- Client evaluation: No data found. ---")
            return 0.0, 0, {"accuracy": 0.0}

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"--- Client evaluation finished: Acc={accuracy:.4f}, Loss={avg_loss:.4f} ---")
        return float(avg_loss), total, {"accuracy": float(accuracy), "loss": float(avg_loss)}
'''
#--------------------------------------------------------------------------------------------
'''
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
import os
from models.hybrid_model import HybridModel 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from pathlib import Path




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
    val_dir = os.path.join(data_path, "Validation") # We now use the Validation folder

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # --- THIS IS THE FIX ---
    # Return both loaders as a tuple
    return train_loader, val_loader
    # --- END OF FIX ---


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # New validation loader

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
            print("--- Client evaluation: No data found. ---")
            return 0.0, 0, {"accuracy": 0.0}

        avg_loss = total_loss / total
        accuracy = correct / total

        current_round = config.get("round", 0)
        if current_round == 5:   # run only at final round
            print("🎯 Generating Grad-CAM visualizations for final round...")
            self.generate_gradcam_visuals()


        print(f"--- Client evaluation finished: Acc={accuracy:.4f}, Loss={avg_loss:.4f} ---")
        return float(avg_loss), total, {"accuracy": float(accuracy), "loss": float(avg_loss)}
    
    
    def generate_gradcam_visuals(self, num_images=50):
        """Generate and save 50 Grad-CAM heatmaps per client in a permanent folder."""
        self.model.eval()

        # Correct layer for EfficientNetV2 (your CNN backbone)
        target_layers = [self.model.cnn_backbone.blocks[-1]]

        cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=(DEVICE.type == 'cuda'))

        # Permanent absolute output directory
        output_dir = Path(r"D:\FedXViT\gradcam_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each client gets its own subfolder (avoid overwriting)
        client_folder = output_dir / f"client_{os.getpid()}"
        client_folder.mkdir(parents=True, exist_ok=True)

        print(f"📁 Grad-CAM outputs for this client will be saved in: {client_folder}")

        count = 0
        for images, _ in self.val_loader:
            for i in range(images.size(0)):
                if count >= num_images:
                    print(f"✅ Saved {count} Grad-CAM images for this client.")
                    return

                # Forward pass one image
                img_tensor = images[i].unsqueeze(0).to(DEVICE)
                grayscale_cam = cam(input_tensor=img_tensor)[0, :]

                # Convert tensor → normalized RGB numpy image
                rgb_img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

                # Overlay heatmap
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # Save Grad-CAM image
                output_path = client_folder / f"img_{count}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                print(f"🖼️ Saved Grad-CAM visualization: {output_path}")
                count += 1
'''
'''
#attention rollout code but with pid as folder name

import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
import os
from models.hybrid_model import HybridModel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Client running on device: {DEVICE}")


# -------------------------------
# 1. Load Client Data
# -------------------------------
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


# -------------------------------
# 2. Beyond Attention Visualization
# -------------------------------
def beyond_attention_visualization(model, val_loader, output_dir, num_images=50):
    """
    Compute Beyond-Attention heatmaps from ViT attention heads for visualization.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    images_to_process = num_images
    processed = 0

    print(f"🎯 Running Beyond-Attention XAI on {num_images} validation images...")

    for images, labels in tqdm(val_loader):
        if processed >= images_to_process:
            break

        images = images.to(DEVICE)
        with torch.no_grad():
            # Forward pass to extract attention weights from ViT
            outputs = model.vit.blocks[-1].attn.get_attn()  # get last layer attention map
            attn = outputs.mean(dim=1).detach().cpu().numpy()  # average over heads

        for i in range(images.size(0)):
            if processed >= images_to_process:
                break

            heatmap = attn[0]  # (num_heads, seq_len, seq_len) reduced → (seq_len, seq_len)
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())

            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.axis("off")
            save_path = os.path.join(output_dir, f"xai_{processed+1:03d}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            processed += 1

    print(f"✅ Beyond-Attention maps saved: {processed}/{num_images} images → {output_dir}")


# -------------------------------
# 3. Flower Client Definition
# -------------------------------
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

        print(f"--- Local Training Done | Last Batch Loss: {last_loss:.4f} ---")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
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

        current_round = config.get("round", 0)
        print(f"--- Evaluation (Round {current_round}) | Acc: {accuracy:.4f}, Loss: {avg_loss:.4f} ---")

        # 🔥 Run Beyond Attention only at round 5
        if current_round == 5:
            print(f"🎯 Round {current_round}: Generating Beyond Attention XAI outputs...")
            output_dir = os.path.join(
                "D:\\FedXViT\\XAI_outputs", f"Client_{os.getpid()}"
            )
            beyond_attention_visualization(self.model, self.val_loader, output_dir, num_images=50)

        return float(avg_loss), total, {"accuracy": float(accuracy), "loss": float(avg_loss)}
'''
# import os
# import torch
# import flwr as fl
# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torch.utils.data import random_split, Dataset
# from tqdm import tqdm

# from models.hybrid_model import HybridModel  # your hybrid ViT+CNN model
# from beyond_attention import beyond_attention_visualize  # custom XAI function you wrote

# # -----------------------------
# # Data Loading Utility
# # -----------------------------
# def load_client_data(client_data_path, batch_size=32, val_split=0.2):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     dataset = torch.utils.data.ImageFolder(client_data_path, transform=transform)
#     val_size = int(val_split * len(dataset))
#     train_size = len(dataset) - val_size

#     train_ds, val_ds = random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader


# # -----------------------------
# # Flower Client
# # -----------------------------
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, val_loader, cid):
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.cid = cid
#         self.current_round = 0  # keep track of round number

#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.tensor(v) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.current_round = config.get("round", self.current_round + 1)
#         print(f"[Client {self.cid}] Starting training for round {self.current_round}")

#         self.model.train()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         criterion = torch.nn.CrossEntropyLoss()

#         for epoch in range(1):
#             for images, labels in tqdm(self.train_loader, desc=f"Client {self.cid} Epoch", leave=False):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 optimizer.zero_grad()
#                 outputs = self.model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#         # ✅ Run XAI (Beyond Attention) ONLY at 5th round
#         if self.current_round == 5:
#             print(f"[Client {self.cid}] Running Beyond Attention visualization on round {self.current_round}")

#             # Create absolute output directory
#             output_dir = os.path.join("D:\\FedXViT\\XAI_outputs", f"Client_{self.cid}")
#             os.makedirs(output_dir, exist_ok=True)

#             beyond_attention_visualize(
#                 model=self.model,
#                 val_loader=self.val_loader,
#                 device=self.device,
#                 output_dir=output_dir,
#                 num_samples=50
#             )
#             print(f"[Client {self.cid}] XAI results saved to {output_dir}")

#         return self.get_parameters(config={}), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         criterion = torch.nn.CrossEntropyLoss()
#         loss, correct = 0.0, 0
#         with torch.no_grad():
#             for images, labels in self.val_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.model(images)
#                 loss += criterion(outputs, labels).item() * images.size(0)
#                 correct += (outputs.argmax(1) == labels).sum().item()
#         loss /= len(self.val_loader.dataset)
#         accuracy = correct / len(self.val_loader.dataset)
#         return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

#     def to_client(self):
#         return self


# # -----------------------------
# # Client Factory Function
# # -----------------------------
# def client_fn(cid: str) -> fl.client.Client:
#     print(f"--- [SERVER LOG] Spawning Client {cid} ---")
#     client_data_path = f"D:\\FedXViT\\SplitData\\CLIENT_{cid}_DATA"
#     print(f"Client {cid} will load data from: {client_data_path}")

#     client_model = HybridModel(num_classes=2)
#     train_loader, val_loader = load_client_data(client_data_path)
#     return FlowerClient(client_model, train_loader, val_loader, cid).to_client()




## 7:15

import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
import os
from models.hybrid_model import HybridModel 

# Set the device for PyTorch (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Client running on device: {DEVICE}")

def load_client_data(data_path):
    """Loads a client's training and validation datasets."""
    print(f"📂 Client loading data from: {data_path}")
    
    # Define transformations for training data (with data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation data (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define paths to the training and validation subfolders
    train_dir = os.path.join(data_path, "Training")
    val_dir = os.path.join(data_path, "Validation")

    # Create PyTorch datasets from the image folders
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)

    # Create data loaders to handle batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Return both the training and validation data loaders
    return train_loader, val_loader

# Define the Flower client class, inheriting from NumPyClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        """Extracts model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Updates the local model with weights received from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """This is the main training function, called by the server."""
        self.set_parameters(parameters) # Update model with latest global weights
        self.model.to(DEVICE)
        self.model.train() # Set the model to training mode

        print(f"--- Client training (1 epoch) on {len(self.train_loader.dataset)} images... ---")
        
        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5) 
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop for one epoch
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
        
        # Return the updated local weights, number of training examples, and an empty dictionary
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """This function evaluates the model, called by the server."""
        print(f"--- Client evaluating on {len(self.val_loader.dataset)} validation images... ---")
        self.set_parameters(parameters) # Update model with the weights to be evaluated
        self.model.to(DEVICE)
        self.model.eval() # Set the model to evaluation mode

        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        # Evaluation loop
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # Handle case where validation set might be empty
        if total == 0:
            print("--- Client evaluation: No data found. ---")
            return 0.0, 0, {"accuracy": 0.0}

        # Calculate average loss and accuracy
        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"--- Client evaluation finished: Acc={accuracy:.4f}, Loss={avg_loss:.4f} ---")
        
        # Return the results back to the server for aggregation
        return float(avg_loss), total, {"accuracy": float(accuracy), "loss": float(avg_loss)}