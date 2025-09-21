# import torch.nn as nn
# import timm
# import torch

# class ViTBackbone(nn.Module):
#     def __init__(self, model_name='vit_base_patch16_224', jft_checkpoint=None):
#         super().__init__()
#         # Initialize ViT model without classification head (num_classes=0)
#         self.vit_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        
#         # Load pretrained JFT or ImageNet weights if checkpoint provided
#         if jft_checkpoint:
#             state_dict = torch.load(jft_checkpoint, map_location='cpu')
#             self.vit_model.load_state_dict(state_dict, strict=False)

#     def forward(self, x):
#         # Forward pass through ViT backbone, output is embedding vector
#         return self.vit_model(x)