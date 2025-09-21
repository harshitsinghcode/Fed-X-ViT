import torch

class AttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model.eval()
        self.discard_ratio = discard_ratio
        self.attention_maps = []

        for blk in self.model.blocks:
            blk.attn.attn_drop.register_forward_hook(self._hook_attention)

    def _hook_attention(self, module, input, output):
        self.attention_maps.append(output.detach())

    def generate(self, input_tensor):
        _ = self.model(input_tensor)  # Forward pass to fill attention_maps

        avg_attn = torch.stack(self.attention_maps).mean(dim=0)[0]  # [num_heads, tokens, tokens]
        avg_attn = avg_attn.mean(dim=0)  # Mean over heads → [tokens, tokens]

        # Discard low attention scores
        flat = avg_attn.flatten()
        val, idx = torch.topk(flat, int(flat.size(0)*self.discard_ratio))
        mask = torch.zeros_like(flat)
        mask[idx] = 1
        mask = mask.reshape(avg_attn.shape)

        # Attention rollout matrix
        attn_rollout = (avg_attn * mask).sum(dim=0)
        
        # Normalize between 0 and 1
        attn_rollout = attn_rollout - attn_rollout.min()
        attn_rollout = attn_rollout / (attn_rollout.max() + 1e-8)
        return attn_rollout.cpu().numpy()
