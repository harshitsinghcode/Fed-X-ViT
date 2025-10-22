import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import functional as F

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        
        # Create hook for features and gradients
        self.feature_maps = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        target_module = dict(self.model.named_modules())[target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        weights = self._compute_weights()
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.feature_maps[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
    
    def _compute_weights(self):
        grads = self.gradients[0]  # Shape: C, H, W
        alpha_num = grads.pow(2)
        alpha_denom = 2*grads.pow(2) + (self.feature_maps[0] * grads.pow(3)).sum(dim=(1,2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * torch.relu(grads)).sum(dim=(1,2))
        return weights
