# D:\FedXViT\models\hybrid_model.py
import torch
import torch.nn as nn
import timm

class HybridModel(nn.Module):
    def __init__(self, num_classes=4, cnn_backbone='tf_efficientnetv2_s.in21k_ft_in1k', vit_model='vit_base_patch16_224.augreg_in21k'):
        super().__init__()
        # This code remains unchanged. It's already perfect for the new task.
        print(f"  - Initializing CNN Backbone ({cnn_backbone})...")
        self.cnn_backbone = timm.create_model(
            cnn_backbone, pretrained=True, features_only=True, out_indices=[3]
        )
        with torch.no_grad():
            dummy_features = self.cnn_backbone(torch.randn(1, 3, 224, 224))
            cnn_feature_dim = dummy_features[0].shape[1]
        print(f"  - Initializing ViT Backbone ({vit_model})...")
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
        print("  - Building projection and classifier layers...")
        self.projection = nn.Conv2d(cnn_feature_dim, self.vit.embed_dim, kernel_size=1)
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)
        print("  - Model components built successfully.")
    def forward(self, x):
        cnn_features = self.cnn_backbone(x)[0]
        projected_features = self.projection(cnn_features)
        tokens = projected_features.flatten(2).transpose(1, 2)
        cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
        tokens_with_cls = torch.cat((cls_token, tokens), dim=1)
        tokens_with_cls = tokens_with_cls + self.vit.pos_embed
        x = self.vit.blocks(tokens_with_cls)
        x = self.vit.norm(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)