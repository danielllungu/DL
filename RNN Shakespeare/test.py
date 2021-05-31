from charRNN import CharRnnDataset
from model_test import CharRNN
import torch
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':
    batch_size = 128
    max_len = 50
    vocab_size = 65

    char_rnn_dataset = CharRnnDataset('shakespeare.txt', max_len)

    net = CharRNN(vocab_size, 256)
    if torch.cuda.is_available():
        net.cuda()

    net.load_state_dict(torch.load('weights/epoch_5', map_location=torch.device('cpu')))

    net.eval()

    idx = 13

    text = char_rnn_dataset.idx_char[idx]

    hx = None
    for i in range(10000):
        inputs = torch.zeros((1, vocab_size))
        inputs[0, idx] = 1

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        outputs, hx = net(inputs, hx)

        probs = F.softmax(outputs, dim=-1).squeeze(0).detach().cpu().numpy()
        chr = np.random.choice(list(char_rnn_dataset.char_idx.keys()), p=probs)
        text += chr
        idx = char_rnn_dataset.char_idx[chr]

    print(text)
