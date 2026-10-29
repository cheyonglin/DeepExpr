import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

##################################  Original Arcface Model #############################################################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# ------------------------ CGA Block ------------------------
# Landmark flow is 49 tokens, Image flow is 1 token
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

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride, attention=None):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))
        
        # Adding the CGA Block (Conditional Gated Attention)
        if attention:
            self.attention = CGAttentionFC(depth)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res = self.attention(res) if hasattr(self, 'attention') else res
        return res + shortcut


class bottleneck_IR_CGA(nn.Module):
    def __init__(self, in_channel, depth, stride, attention=None):
        super(bottleneck_IR_CGA, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )
        
        # Adding the CGA Block (Conditional Gated Attention)
        if attention:
            self.attention = CGAttentionFC(depth)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res = self.attention(res) if hasattr(self, 'attention') else res
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2, attention=False):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers, attention=False):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, attention=attention),
            get_block(in_channel=64, depth=128, num_units=4, attention=attention),
            get_block(in_channel=128, depth=256, num_units=14, attention=attention)
        ]
        
    elif num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2, attention=attention),
            get_block(in_channel=64, depth=128, num_units=2, attention=attention),
            get_block(in_channel=128, depth=256, num_units=2, attention=attention)  # 修改了这里的num_units
        ]
        
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, attention=attention),
            get_block(in_channel=64, depth=128, num_units=4, attention=attention),
            get_block(in_channel=128, depth=256, num_units=6, attention=attention)  # 修改了这里的num_units
        ]
    elif num_layers == 40:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, attention=attention),
            get_block(in_channel=64, depth=128, num_units=4, attention=attention),
            get_block(in_channel=128, depth=256, num_units=8, attention=attention)
        ]
        
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, attention=attention),
            get_block(in_channel=64, depth=128, num_units=13, attention=attention),
            get_block(in_channel=128, depth=256, num_units=30, attention=attention),
            #get_block(in_channel=256, depth=512, num_units=3, attention=attention)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3, attention=attention),
            get_block(in_channel=64, depth=128, num_units=8, attention=attention),
            get_block(in_channel=128, depth=256, num_units=36, attention=attention),
            #get_block(in_channel=256, depth=512, num_units=3, attention=attention)
        ]
    return blocks

# -------------- Backbone 模型 --------------
class Backbone(nn.Module):
    def __init__(self, num_layers=34, drop_ratio=0.0, mode='ir', out_dim=512, attention=False):
        super().__init__()
        assert num_layers in [18, 34, 40, 50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        # 配置每个层的 block
        blocks = get_blocks(num_layers, attention=attention)
        unit_module = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                    attention=attention
                ))
        self.body = nn.Sequential(*modules)

        # 自适应输出层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 计算输出层输入的深度
        last_block_depth = blocks[-1][-1].depth  # 获取最后一个 block 的深度
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_block_depth, out_dim),  # out_dim 可指定，比如 512
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = F.interpolate(x, size=112)  # Resizing input image
        x = self.input_layer(x)
        x = self.body(x)
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = self.output_layer(x)  # (B, out_dim)
        return x

# -------------- 测试代码 --------------
if __name__ == "__main__":
    # 测试 IR50 模型
    model50 = Backbone(num_layers=50, out_dim=512, attention=True)
    dummy = torch.randn(1, 3, 112, 112)
    out34 = model50(dummy)
    print("IR34 Output shape:", out34.shape)  # torch.Size([1, 512])
