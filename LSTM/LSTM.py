# ----------------------------------------------------#
#   LSTM 模型
# ----------------------------------------------------#
class LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=256, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_size)  # output_size classes

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        x = self.fc(x)  # 通过一个全连接层
        return x
