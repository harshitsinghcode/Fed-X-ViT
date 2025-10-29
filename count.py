import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.hybrid_model import HybridModel
import mlflow

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # Use full backward hook

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(pooled_grads.size(0)):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        return heatmap.numpy()

def save_heatmap_on_image(original_image, heatmap, save_path, alpha=0.4):
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + original_image
    cv2.imwrite(save_path, np.uint8(superimposed_img))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model trained after federated rounds
    model = HybridModel(num_classes=2)
    model.load_state_dict(torch.load("final_federated_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Target the last CNN block (adjust if different)
    target_layer = model.cnn_backbone.blocks[3]
    gradcam = GradCAM(model, target_layer)

    # Test dataset path
    dataset_path = "D:/FedXViT/SplitData/CLIENT_0_DATA/Testing"  # Change as needed

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = "outputs/gradcam_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # Start MLflow run for logging
    with mlflow.start_run(run_name="GradCAM_Heatmaps") as run:
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            heatmap = gradcam.generate_heatmap(inputs)

            # Denormalize image for overlay
            img_tensor = inputs[0].cpu()
            mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            std = np.array([0.229, 0.224, 0.225])[None, None, :]
            img = img_tensor.numpy().transpose(1, 2, 0) * std + mean
            img = np.clip(img, 0, 1)
            img = np.uint8(255 * img)

            save_path = os.path.join(output_dir, f"gradcam_{idx}.jpg")
            save_heatmap_on_image(img, heatmap, save_path)

        # Log all heatmaps as artifacts
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".jpg"):
                mlflow.log_artifact(os.path.join(output_dir, file_name), artifact_path="gradcam_heatmaps")
        # Tag the run as successful
        mlflow.set_tag("GradCAM_Status", "Successful")

    print(f"Heatmaps are saved in {output_dir} and logged in Azure ML MLflow artifacts.")

if __name__ == "__main__":
    main()
