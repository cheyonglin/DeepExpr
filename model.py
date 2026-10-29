import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilefacenet import MobileFaceNet
from IR import Backbone
from hyp_crossvit_affect import HyVisionTransformer

# ------------------------
# CGA Block
# ------------------------
class CGAttentionFC(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)
# ------------------------
# Classification Head
# ------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)



# ------------------------
# Pyramid Transformer Expression Model
# ------------------------
class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=6, type=6):
        """
        Args:
            img_size: 输入图像尺寸
            num_classes: 分类类别数
            type: 控制 Transformer 的深度，范围 1~8
        """
        super().__init__()

        assert 1 <= type <= 8, f"type 必须在 1~8 之间，你传入的是 {type}"

        self.img_size = img_size
        self.num_classes = num_classes
        self.depth = type  # 直接用 type 作为 depth

        print(f"[INFO] 初始化 Pyramid Transformer Expression Model，depth={self.depth}")

        # MobileFaceNet: 112x112 输出 136 landmarks + embedding
        self.face_landback = MobileFaceNet([112, 112], 136)

        # IR50 backbone 输出 512-d feature
        self.ir_back = Backbone(50, drop_ratio=0.0, mode='ir', out_dim=512)
        self.ir_layer = nn.Linear(512, 512)  # 保证与 transformer embedding 一致

        # Pyramid transformer
        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=self.depth,       # 用 self.depth
            num_heads=8, mlp_ratio=2.0,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )

        self.cga_block = CGAttentionFC(512)
        self.head = ClassificationHead(512, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]

        # --- Landmark features ---
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)  # 只取 embedding
        x_face = x_face.view(B, -1, 49).transpose(1, 2)  # (B, 49, embedding_dim)

        # --- Image features ---
        x_ir = self.ir_back(x)  # (B, 512)
        x_ir = self.ir_layer(x_ir).unsqueeze(1)  # (B, 1, 512)

        # --- Fuse ---
        y_hat = self.pyramid_fuse(x_ir, x_face)  # (B, 512)
        y_hat = self.cga_block(y_hat)
        out = self.head(y_hat)
        return out, y_hat


# ------------------------
# 使用示例
# ------------------------
if __name__ == "__main__":
    model1 = pyramid_trans_expr(type=1)  # depth=1
    model4 = pyramid_trans_expr(type=4)  # depth=4
    model8 = pyramid_trans_expr(type=8)  # depth=8

    dummy_input = torch.randn(2, 3, 224, 224)
    out, feat = model4(dummy_input)
    print("Output shape:", out.shape)   # (2, num_classes)
    print("Feature shape:", feat.shape) # (2, 512)
