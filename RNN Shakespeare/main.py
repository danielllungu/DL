from charRNN import CharRnnDataset
from torch.utils.data import DataLoader
from model import CharRNN
import torch
import torch.optim as optim
import torch.nn as nn


if __name__ == '__main__':
    batch_size = 128
    max_len = 50
    vocab_size = 65

    char_rnn_dataset = CharRnnDataset('shakespeare.txt', max_len)
    train_loader = DataLoader(char_rnn_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = CharRNN(vocab_size, 256)
    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        for idx, (inputs, targets) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print('Loss at epoch {}, {}/{}: {}'.format(epoch + 1, idx, len(train_loader), loss.item()))

        torch.save(net.state_dict(), 'weights/epoch_{}'.format(epoch + 1))
