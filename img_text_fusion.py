import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.text_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.image_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, t):
        B, N, C = x.shape
        B, N_text, C_text = t.shape

        text_qkv = (
            self.text_qkv(t)
            .reshape(B, N_text, 3, self.num_heads, C_text // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = text_qkv[0]

        image_qkv = (
            self.image_qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = image_qkv[1], image_qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fused_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.image_proj = nn.Conv2d(image_dim, fused_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads)
        self.output_proj = nn.Conv2d(fused_dim, image_dim, kernel_size=1)

    def forward(self, text_features, image_features):
        # (32, 77, 512) -> (32, 77, fused_dim)
        text_features = self.text_proj(text_features)

        # (32, 56, 56, 96) -> (32, 96, 56, 56)
        image_features = image_features.permute(0, 3, 1, 2)
        batch_size, _, height, width = image_features.size()

        # (32, 96, 56, 56) -> (32, fused_dim, 56, 56)
        image_features = self.image_proj(image_features)

        # (32, fused_dim, 56 * 56) -> (32, 56 * 56, fused_dim)
        image_features_flat = image_features.view(batch_size, fused_dim, height * width).permute(0, 2, 1)

        # Cross Attention
        fused_features, _ = self.attention(image_features_flat, text_features, text_features)

        # (32, 56 * 56, fused_dim) -> (32, fused_dim, 56, 56)
        fused_features = fused_features.permute(0, 2, 1).view(batch_size, fused_dim, height, width)

        # (32, fused_dim, 56, 56) -> (32, image_dim, 56, 56)
        fused_features = self.output_proj(fused_features)

        # (32, image_dim, 56, 56) -> (32, 56, 56, image_dim)
        fused_features = fused_features.permute(0, 2, 3, 1)
        return fused_features



text_features = torch.randn(32, 77, 512)  # (batch_size, sequence_length, feature_dim)
image_features = torch.randn(32, 56, 56, 96)  # (batch_size, height, width, feature_dim)


fused_dim = 128
num_heads = 4
fusion_module = CrossAttentionFusion(text_dim=512, image_dim=96, fused_dim=fused_dim, num_heads=num_heads)


fused_features = fusion_module(text_features, image_features)
print(fused_features.shape)  # (32, 56, 56, 96)
