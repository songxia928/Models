import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorProductAttention(nn.Module):
    def __init__(
        self,
        d_model: int,        # 输入维度
        num_heads: int,      # 注意力头数
        d_k: int,            # 每个头的维度
        r_q: int = 6,        # 查询分解秩
        r_k: int = 2,        # 键分解秩
        r_v: int = 2         # 值分解秩
    ):
        super(TensorProductAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.r_q, self.r_k, self.r_v = r_q, r_k, r_v

        # 分解矩阵：头维度 (h) 和 令牌维度 (d_k) 的线性层
        self.W_a_q = nn.Linear(d_model, num_heads * r_q)     # 头维度因子 (A_Q)
        self.W_b_q = nn.Linear(d_model, d_k * r_q)          # 令牌维度因子 (B_Q)
        self.W_a_k = nn.Linear(d_model, num_heads * r_k)     # A_K
        self.W_b_k = nn.Linear(d_model, d_k * r_k)          # B_K
        self.W_a_v = nn.Linear(d_model, num_heads * r_v)     # A_V
        self.W_b_v = nn.Linear(d_model, d_k * r_v)          # B_V

        self.out_proj = nn.Linear(num_heads * d_k, d_model)  # 输出投影层

    def forward(self, x: torch.Tensor, rope_pos: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        # 生成分解因子 (A, B)
        # Q: (batch, seq, r_q*h) -> (batch, seq, r_q, h)
        A_Q = self.W_a_q(x).view(batch_size, seq_len, self.r_q, self.num_heads).transpose(1, 2)
        B_Q = self.W_b_q(x).view(batch_size, seq_len, self.r_q, self.d_k).transpose(1, 2)
        # K: (batch, seq, r_k*h) -> (batch, seq, r_k, h)
        A_K = self.W_a_k(x).view(batch_size, seq_len, self.r_k, self.num_heads).transpose(1, 2)
        B_K = self.W_b_k(x).view(batch_size, seq_len, self.r_k, self.d_k).transpose(1, 2)
        # V: (batch, seq, r_v*h) -> (batch, seq, r_v, h)
        A_V = self.W_a_v(x).view(batch_size, seq_len, self.r_v, self.num_heads).transpose(1, 2)
        B_V = self.W_b_v(x).view(batch_size, seq_len, self.r_v, self.d_k).transpose(1, 2)

        # 应用 RoPE（假设 rope_pos 为位置相关的旋转矩阵）
        if rope_pos is not None:
            # 调整 rope_pos 的形状以匹配 B_Q 和 B_K
            rope_pos = rope_pos.unsqueeze(0).unsqueeze(1)  # 形状变为 (1, 1, seq_len, d_k)
            B_Q = B_Q * rope_pos
            B_K = B_K * rope_pos

        # 重构 Q, K, V
        Q = (A_Q.unsqueeze(-1) * B_Q.unsqueeze(-2)).sum(dim=1)  # (batch, seq, h, d_k)
        K = (A_K.unsqueeze(-1) * B_K.unsqueeze(-2)).sum(dim=1)  # (batch, seq, h, d_k)
        V = (A_V.unsqueeze(-1) * B_V.unsqueeze(-2)).sum(dim=1)  # (batch, seq, h, d_k)

        # 注意力计算：(batch, seq, h, d_k) @ (batch, seq, d_k, h) -> (batch, h, seq, seq)
        scores = (Q @ K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = (attn_weights @ V).transpose(1, 2).contiguous()  # (batch, seq, h, d_k)

        # 拼接头并投影
        output = output.view(batch_size, seq_len, -1)
        return self.out_proj(output)




# 超参数
d_model = 768       # 输入维度（如 BERT-base）
num_heads = 12      # 注意力头数
d_k = 64            # 每个头的维度
r_q, r_k, r_v = 6, 2, 2  # 分解秩（论文默认配置）
batch_size, seq_len = 2, 1024  # 批次大小和序列长度

# 输入数据：(batch_size, seq_len, d_model)
x = torch.randn(batch_size, seq_len, d_model)
# RoPE 位置编码（示例：简化的正弦编码，实际需匹配模型）
rope_pos = torch.randn(seq_len, d_k)

tpa = TensorProductAttention(d_model, num_heads, d_k, r_q, r_k, r_v)
output = tpa(x, rope_pos)

# 验证输出形状：(batch_size, seq_len, d_model)
print(f"Output shape: {output.shape}")  # 应输出 (2, 1024, 768)