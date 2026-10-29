import torch
from torch import nn
import torch.nn.functional as F

# -------------------- Drop Path --------------------
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# -------------------- Attention --------------------
class Attention_img(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_img, x_lm):
        cls_img, x_img_tokens = x_img[:, :1, :], x_img[:, 1:, :]
        cls_lm, x_lm_tokens = x_lm[:, :1, :], x_lm[:, 1:, :]

        B, N_img, C = x_img_tokens.shape
        B, N_lm, C = x_lm_tokens.shape

        kv = self.kv(x_img_tokens).reshape(B, N_img, 2, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        k, v = kv.unbind(0)
        q = x_lm_tokens.reshape(B, N_lm, self.num_heads, C//self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out_tokens = (attn @ v).transpose(1,2).reshape(B, N_lm, C)
        x_out_tokens = self.proj(x_out_tokens)
        x_out_tokens = self.proj_drop(x_out_tokens)

        x_out = torch.cat([cls_img, x_out_tokens], dim=1)
        return x_out

class Attention_lm(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_lm, x_img):
        cls_lm, x_lm_tokens = x_lm[:, :1, :], x_lm[:, 1:, :]
        cls_img, x_img_tokens = x_img[:, :1, :], x_img[:, 1:, :]

        B, N_lm, C = x_lm_tokens.shape
        B, N_img, C = x_img_tokens.shape

        kv = self.kv(x_lm_tokens).reshape(B, N_lm, 2, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        k, v = kv.unbind(0)
        q = x_img_tokens.reshape(B, N_img, self.num_heads, C//self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out_tokens = (attn @ v).transpose(1,2).reshape(B, N_img, C)
        x_out_tokens = self.proj(x_out_tokens)
        x_out_tokens = self.proj_drop(x_out_tokens)

        x_out = torch.cat([cls_img, x_out_tokens], dim=1)
        return x_out

# -------------------- MLP --------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# -------------------- Transformer Block --------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm_img = nn.LayerNorm(dim)
        self.norm_lm = nn.LayerNorm(dim)
        self.attn_img = Attention_img(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.attn_lm = Attention_lm(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden = int(dim*mlp_ratio)
        self.mlp_img = Mlp(dim, mlp_hidden, drop=drop)
        self.mlp_lm = Mlp(dim, mlp_hidden, drop=drop)

    def forward(self, x_img, x_lm):
        x_img_tokens = x_img[:, 1:, :]
        x_lm_tokens = x_lm[:, 1:, :]

        x_img_tokens = x_img_tokens + self.drop_path(self.attn_img(torch.cat([x_img[:, :1, :], x_img_tokens], dim=1),
                                                                    torch.cat([x_lm[:, :1, :], x_lm_tokens], dim=1))[:, 1:, :])
        x_img_tokens = x_img_tokens + self.drop_path(self.mlp_img(self.norm_img(x_img_tokens)))
        x_lm_tokens = x_lm_tokens + self.drop_path(self.attn_lm(torch.cat([x_lm[:, :1, :], x_lm_tokens], dim=1),
                                                                torch.cat([x_img[:, :1, :], x_img_tokens], dim=1))[:, 1:, :])
        x_lm_tokens = x_lm_tokens + self.drop_path(self.mlp_lm(self.norm_lm(x_lm_tokens)))

        x_img = torch.cat([x_img[:, :1, :], x_img_tokens], dim=1)
        x_lm = torch.cat([x_lm[:, :1, :], x_lm_tokens], dim=1)
        return x_img, x_lm

# -------------------- Pyramid Block --------------------
class PyramidBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., depth=1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio, drop, attn_drop, drop_path) for _ in range(depth)])
        self.proj_down = nn.Linear(dim, dim//2)
        self.proj_up = nn.Linear(dim//2, dim)

    def forward(self, x_img, x_lm):
        for blk in self.blocks:
            x_img, x_lm = blk(x_img, x_lm)
        x_down = self.proj_down(x_img[:, 1:, :])
        x_up = self.proj_up(x_down)
        x_img = torch.cat([x_img[:, :1, :], x_img[:, 1:, :] + x_up], dim=1)
        return x_img, x_lm

# -------------------- Hybrid Vision Transformer --------------------
class HyVisionTransformer(nn.Module):
    def __init__(self, in_chans=49, q_chanel=49, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans + q_chanel + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.pyramid = nn.ModuleList([PyramidBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_img, x_lm):
        B = x_img.shape[0]
        cls_img = torch.mean(x_img,1,keepdim=True)
        cls_lm = torch.mean(x_lm,1,keepdim=True)
        x_img = torch.cat([cls_img, x_img], dim=1)
        x_lm = torch.cat([cls_lm, x_lm], dim=1)

        # pos_embed
        x_img = x_img + self.pos_embed[:, :x_img.shape[1], :]
        x_lm = x_lm + self.pos_embed[:, x_img.shape[1]:x_img.shape[1]+x_lm.shape[1], :]
        x_img = self.pos_drop(x_img)
        x_lm = self.pos_drop(x_lm)

        for block in self.pyramid:
            x_img, x_lm = block(x_img, x_lm)

        x_img = self.norm(x_img)
        return x_img[:,0,:]
