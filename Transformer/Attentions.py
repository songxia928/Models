import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim, d_k):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, d_k)
        self.W_k = nn.Linear(input_dim, d_k)
        self.W_v = nn.Linear(input_dim, d_k)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1)).float())
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output



# 1.2 MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, d_k):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.head_dim = d_k // num_heads

        # 线性层用于生成 Q, K, V
        self.W_q = nn.Linear(input_dim, d_k)
        self.W_k = nn.Linear(input_dim, d_k)
        self.W_v = nn.Linear(input_dim, d_k)
        self.W_o = nn.Linear(d_k, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn_weights = F.softmax(scores, dim=-1)

        # 计算加权和
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.W_o(output)
        return output




# 2. Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 3. Temporal Attention
class TemporalAttention(nn.Module):
    def __init__(self, input_dim, d_k, time_emb_dim):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim  # 输入文本嵌入维度
        self.d_k = d_k  # Q/K/V维度
        self.time_emb_dim = time_emb_dim  # 时间嵌入维度
        
        # 文本的Q/K/V线性层
        self.W_q = nn.Linear(input_dim, d_k)
        self.W_k = nn.Linear(input_dim, d_k)
        self.W_v = nn.Linear(input_dim, d_k)
        
        # 时间嵌入的线性层（将时间点映射到d_k维度）
        self.W_t = nn.Linear(time_emb_dim, d_k)
    
    def forward(self, x, time_points):
        batch_size, seq_len, _ = x.size()
        num_time_points = time_points.size(1)  # 假设每个序列对应一个时间点
        
        # 生成文本的Q/K/V
        Q = self.W_q(x)  # (batch_size, seq_len, d_k)
        K = self.W_k(x)  # (batch_size, seq_len, d_k)
        V = self.W_v(x)  # (batch_size, seq_len, d_k)
        
        # 生成时间矩阵T：将时间点嵌入扩展为序列长度一致的矩阵
        time_emb = self.W_t(time_points)  # (batch_size, 1, d_k)
        T = time_emb.repeat(1, seq_len, 1)  # (batch_size, seq_len, d_k)
        
        # 计算时间矩阵的全局范数（Frobenius范数）
        T_norm = torch.norm(T, p='fro', dim=(1, 2), keepdim=True)  # (batch_size, 1, 1)
        T_scaled = (T.transpose(1, 2) @ T) / (T_norm + 1e-8)  # (batch_size, d_k, d_k)
                                                             # a @ b 就相当于 torch.matmul(a, b)
        
        # 计算注意力分数：Q * T_scaled * K^T
        scores = torch.matmul(Q, T_scaled)  # (batch_size, seq_len, d_k)
        scores = torch.matmul(scores, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 归一化和加权求和
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights



# 4. Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, input_dim, d_k):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, d_k)
        self.W_k = nn.Linear(input_dim, d_k)
        self.W_v = nn.Linear(input_dim, d_k)

    def forward(self, x1, x2):
        Q = self.W_q(x1)
        K = self.W_k(x2)
        V = self.W_v(x2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1)).float())
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


# 5. Grouped Attention
class GroupedAttention(nn.Module):
    def __init__(self, input_dim, d_k, num_heads, num_groups):
        super(GroupedAttention, self).__init__()
        assert num_heads % num_groups == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_k // num_heads
        self.W_q = nn.Linear(input_dim, d_k)
        self.W_k = nn.Linear(input_dim, d_k // num_groups)
        self.W_v = nn.Linear(input_dim, d_k // num_groups)
        self.W_o = nn.Linear(d_k, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads // self.num_groups, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads // self.num_groups, self.head_dim).transpose(1, 2)
        # 调整 Q 的形状以匹配 K 的分组
        Q = Q.view(batch_size, self.num_groups, self.num_heads // self.num_groups, seq_len, self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        return output


# 6. Tensor Product Attention
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



# 测试代码
if __name__ == "__main__":
    # ======== 测试 Self-Attention
    input_dim = 64
    d_k = 32
    seq_len = 10
    batch_size = 2

    x = torch.randn(batch_size, seq_len, input_dim)
    self_attn = SelfAttention(input_dim, d_k)
    output_self = self_attn(x)
    print("Self-Attention output shape:", output_self.shape)


    # ======== 测试 MHA
    num_heads = 4
    input_dim = 64
    d_k = 32
    seq_len = 10
    batch_size = 2

    x = torch.randn(batch_size, seq_len, input_dim)
    mha = MultiHeadAttention(input_dim, num_heads, d_k)
    output = mha(x)
    print("Multi-Head Attention output shape:", output.shape)
    

    # ======== 测试 Spatial Attention
    channels = 3
    height = 16
    width = 16

    x_spatial = torch.randn(batch_size, channels, height, width)

    spatial_attn = SpatialAttention()
    attn_map = spatial_attn(x_spatial)
    output_spatial = x_spatial * attn_map
    print("Spatial-Attention output shape:", output_spatial.shape)

    # ======== 测试 Temporal Attention
    # 超参数
    input_dim = 768  # BERT词嵌入维度
    d_k = 64  # Q/K/V维度
    time_emb_dim = 32  # 时间嵌入维度
    batch_size = 2
    seq_len = 10
    
    # 输入：文本嵌入和时间点（每个序列对应一个时间点，维度为(batch_size, 1, time_emb_dim)）
    x = torch.randn(batch_size, seq_len, input_dim)
    time_points = torch.randn(batch_size, 1, time_emb_dim)  # 例如时间戳的嵌入

    temporal_attn = TemporalAttention(input_dim, d_k, time_emb_dim)
    output_temporal, attn_weights = temporal_attn(x, time_points)
    print("Temporal-Attention output shape:", output_temporal.shape) # 应输出(batch_size, seq_len, d_k)

    # ======== 测试 Cross-Attention
    x1 = torch.randn(batch_size, seq_len, input_dim)
    x2 = torch.randn(batch_size, seq_len, input_dim)
    cross_attn = CrossAttention(input_dim, d_k)
    output_cross = cross_attn(x1, x2)
    print("Cross-Attention output shape:", output_cross.shape)

    # ======== 测试 Grouped Attention
    num_heads = 4
    num_groups = 2
    grouped_attn = GroupedAttention(input_dim, d_k, num_heads, num_groups)
    output_grouped = grouped_attn(x)
    print("Grouped-Attention output shape:", output_grouped.shape)


    # ======== 测试 Tensor Product Attention
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
    output_tpa = tpa(x, rope_pos)

    # 验证输出形状：(batch_size, seq_len, d_model)
    print("Tensor Product Attention output shape:", output_tpa.shape)  # 应输出 (2, 1024, 768)


