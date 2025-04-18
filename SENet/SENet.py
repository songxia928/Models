import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层
            nn.Sigmoid()  # Sigmoid 函数生成权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的 batch size 和通道数
        y = self.avg_pool(x).view(b, c)  # 全局平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层生成权重并调整维度
        return x * y.expand_as(x)  # 对每个通道加权后返回

# 测试
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # 假设输入为 (batch_size=8, channels=64, height=32, width=32)
    se_layer = SELayer(channel=64, reduction=16)
    output_tensor = se_layer(input_tensor)
    print(output_tensor.shape)  # 输出形状和输入一致
