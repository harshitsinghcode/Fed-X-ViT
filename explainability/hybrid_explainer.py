import torch
from gradcam_plus import GradCAMPlusPlus
from attention_rollout import AttentionRollout

class HybridExplainer:
    def __init__(self, model, cnn_target_layer='features.6', discard_ratio=0.9):
        # cnn_target_layer depends on resnet backbone feature layer; adjust as needed
        self.model = model
        self.gradcam = GradCAMPlusPlus(self.model.resnet, cnn_target_layer)
        self.att_rollout = AttentionRollout(self.model.vit, discard_ratio=discard_ratio)

    def explain(self, input_tensor, class_idx=None):
        self.model.eval()
        with torch.no_grad():
            cam_map = self.gradcam.generate(input_tensor, class_idx)
            attn_map = self.att_rollout.generate(input_tensor)
        return cam_map, attn_map
