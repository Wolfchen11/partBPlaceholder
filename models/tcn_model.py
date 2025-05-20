# models/tcn_model.py

import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_size, input_size, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return self.relu(x + res)

class TCNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, seq_len=96, output_size=1):
        super(TCNModel, self).__init__()
        self.block = TCNBlock(input_size, hidden_size)
        self.linear = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # to (batch, channels, seq_len)
        x = self.block(x)
        x = x.mean(dim=1)
        return self.linear(x)