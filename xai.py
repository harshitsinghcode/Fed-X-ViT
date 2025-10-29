import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import random # [MODIFIED] Import the random module

# Ensure you have installed this library: pip install opencv-python grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("🚨 Error: 'grad-cam' or 'opencv-python' is not installed.")
    print("Please install them by running: pip install opencv-python grad-cam")
    exit()

# --- 1. IMPORT YOUR MODEL DEFINITION ---
from models.hybrid_model import HybridModel

# --- 2. CONFIGURATION ---
MODEL_PATH = "final_federated_model.pth"
BASE_DATA_PATH = r"D:\FedXViT\SplitData"
CLIENT_IDS = [0, 1, 2]
NUM_IMAGES_PER_CLIENT = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_image(pil_image):
    """Prepares a single image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_image).unsqueeze(0)

def generate_heatmaps_for_all_clients():
    """
    Main function to load the model, loop through all clients,
    run Grad-CAM on 20 random test images, and save visualizations.
    """
    print(f"--- 🧠 Starting XAI Heatmap Generation ({NUM_IMAGES_PER_CLIENT} random images each) ---")

    # --- 3. LOAD THE MODEL ONCE ---
    print(f"🧠 Loading model from: {MODEL_PATH}")
    model = HybridModel(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"🚨 FATAL ERROR: Model file not found at '{MODEL_PATH}'.")
        return
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 4. INITIALIZE Grad-CAM ONCE ---
    target_layers = [model.vit.blocks[-1].norm1]
    print(f"🎯 Target layer for Grad-CAM set to the last block's normalization layer.")

    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # --- 5. LOOP THROUGH EACH CLIENT ---
    for client_id in CLIENT_IDS:
        print(f"\n--- Processing CLIENT {client_id} ---")

        # Define paths for this specific client
        client_dir = os.path.join(BASE_DATA_PATH, f"CLIENT_{client_id}_DATA")
        test_tumor_dir = os.path.join(client_dir, "Testing", "Tumor")
        heatmap_output_dir = os.path.join(client_dir, "HeatMaps_10")

        # Create the HeatMaps directory if it doesn't exist
        os.makedirs(heatmap_output_dir, exist_ok=True)
        print(f"  Saving heatmaps to: {heatmap_output_dir}")

        # Find all images (jpg, JPG, png) in the client's test tumor folder
        image_paths = []
        for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
            image_paths.extend(glob.glob(os.path.join(test_tumor_dir, ext)))

        if not image_paths:
            print(f"  ⚠️ WARNING: No tumor images found for CLIENT {client_id} in {test_tumor_dir}")
            continue # Skip to the next client

        # [MODIFIED] Shuffle the list of images to get a random order
        random.shuffle(image_paths)

        # Select the first 20 images from the now-randomized list
        images_to_process = image_paths[:NUM_IMAGES_PER_CLIENT]
        print(f"  📂 Found {len(image_paths)} images. Processing {len(images_to_process)} random images.")

        # --- 6. PROCESS IMAGES FOR THIS CLIENT ---
        for i, img_path in enumerate(images_to_process):
            try:
                # Print progress every 5 images
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"    - Processing image {i+1}/{len(images_to_process)}: {os.path.basename(img_path)}")

                pil_image = Image.open(img_path).convert("RGB")
                input_tensor = preprocess_image(pil_image).to(DEVICE)

                rgb_img = np.array(pil_image.resize((224, 224))) / 255.0

                targets = [ClassifierOutputTarget(1)] # Target class is "Tumor"

                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                output_filename = f"HeatMap_{os.path.basename(img_path)}"
                output_path = os.path.join(heatmap_output_dir, output_filename)

                # Save the image with the correct color conversion
                cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"    - 🚨 Failed to process {os.path.basename(img_path)}. Error: {e}")

        print(f"  ✅ Finished processing CLIENT {client_id}.")

    print("\n--- ✅ All clients processed. XAI Generation Complete. ---")


if __name__ == "__main__":
    generate_heatmaps_for_all_clients()