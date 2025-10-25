# D:\FedXViT\xai_test.py
import torch
import glob
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

# --- 1. Import your model and the XAI tools ---
from models.hybrid_model import HybridModel
from captum.attr import GradCAM
from captum.img.viz import visualize_image_attr

# --- 2. Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "final_federated_model.pth" # The model we are testing
DATA_PATH = r"D:\FedXViT\SplitData\CLIENT_0_DATA\Testing" # Path to sample images

# --- 3. Helper function to load and process one image ---
def load_image(img_path):
    """Loads and transforms a single image for the model."""
    # Load the raw image for plotting
    img = Image.open(img_path).convert("RGB")
    
    # Transformation for the "raw" plot
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Transformation for the model
    model_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_img_tensor = raw_transform(img)
    input_tensor = model_transform(raw_img_tensor).unsqueeze(0) # Add batch dim
    return input_tensor.to(DEVICE), raw_img_tensor

# --- 4. Main XAI Function ---
if __name__ == "__main__":
    print(f"--- [XAI ANALYSIS SCRIPT] ---")
    print(f"Running on device: {DEVICE}")
    print(f"Loading model: {MODEL_PATH}")

    # --- Load the model ---
    model = HybridModel(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"🚨 Error: Could not load model. Did you run the server script first?")
        print(f"Details: {e}")
        exit()
        
    model.to(DEVICE)
    model.eval()

    # --- Select the layer for Grad-CAM ---
    # We want the last convolutional layer. In your model, it's 'cnn_backbone.blocks[6]'
    try:
        target_layer = model.cnn_backbone.blocks[6]
        grad_cam = GradCAM(model, target_layer)
    except Exception as e:
        print(f"Error selecting Grad-CAM layer: {e}. Check your model architecture.")
        exit()

    # --- Find sample images to test ---
    # Class 0 = Healthy, Class 1 = Tumor
    class_names = ['Healthy', 'Tumor']
    sample_paths = [
        (os.path.join(DATA_PATH, "Healthy", "Hclient1 (1).jpg"), 0, "healthy_sample"),
        (os.path.join(DATA_PATH, "Tumor", "Tclient1 (1).jpg"), 1, "tumor_sample")
    ]
    
    print("\n--- Starting Grad-CAM Analysis ---")
    
    for img_path, target_class, filename in sample_paths:
        print(f"Processing: {filename} (Target: {class_names[target_class]})...")
        
        try:
            input_tensor, raw_img_tensor = load_image(img_path)
        except Exception as e:
            print(f"  Could not load image {img_path}. Skipping.")
            continue
            
        # --- Run Grad-CAM ---
        # Generate the explanation (attribution) for the target class
        attribution = grad_cam.attribute(input_tensor, target=target_class)
        
        # --- Visualize and Save ---
        img_np = raw_img_tensor.permute(1, 2, 0).numpy() # (H, W, C)
        attr_np = attribution.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() # (H, W, C)
        
        fig, _ = visualize_image_attr(
            attr_np,
            img_np,
            method="blended_heat_map", # You can also try "heat_map"
            sign="all",
            show_colorbar=True,
            title=f"XAI for {class_names[target_class]} Prediction"
        )
        
        save_path = f"xai_{filename}.png"
        fig.savefig(save_path)
        print(f"✅ Saved heatmap to {save_path}")

    print("\n--- [XAI ANALYSIS COMPLETE] ---")