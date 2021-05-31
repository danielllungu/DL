import torch.nn as nn
from torch.nn import Module

class CharRNN(Module):
    def __init__(self, input_size, hidden_size):
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.LSTMCell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hx):
        hx = self.LSTMCell(x, hx)
        x = self.fc(hx[0])
        return x, hx