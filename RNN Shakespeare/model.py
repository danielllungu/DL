import torch.nn as nn
from torch.nn import Module
import torch

class CharRNN(Module):
    def __init__(self, input_size, hidden_size):
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.LSTMCell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        outputs = list()
        hx = None
        for i in range(x.size(1)):
            hx = self.LSTMCell(x[:, i], hx)
            outputs.append(hx[0].unsqueeze(1))

        x = torch.cat(outputs, dim=1)
        x = self.fc(x)
        return x