# import torch
# from torch import nn
# from transformers import ViTModel, ViTImageProcessor

# class ViTBackbone(nn.Module):
#     def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
#         super().__init__()
#         # Load pretrained ViT (without classification head)
#         self.vit = ViTModel.from_pretrained(model_name)
#         # Load corresponding image processor for preprocessing outside this class
#         self.processor = ViTImageProcessor.from_pretrained(model_name)

#     def forward(self, images):
#         """
#         images: Tensor of shape [B, 3, H, W], normalized appropriately
#         Returns: CLS token embeddings of shape [B, hidden_size]
#         """
#         outputs = self.vit(images)
#         # CLS token embedding is at position 0
#         cls_embeddings = outputs.last_hidden_state[:, 0]
#         return cls_embeddings
