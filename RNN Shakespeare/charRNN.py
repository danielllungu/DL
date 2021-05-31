from torch.utils.data import Dataset
import torch


class CharRnnDataset(Dataset):
    def __init__(self, path, max_len=50):
        with open(path, 'r') as f:
            self.dataset = f.read()

        idx = 0
        self.char_idx = {}
        self.idx_char = {}
        for i in range(len(self.dataset)):
            if self.dataset[i] not in self.char_idx.keys():
                self.char_idx[self.dataset[i]] = idx
                self.idx_char[idx] = self.dataset[i]
                idx += 1

        self.max_len = max_len

    def __getitem__(self, idx):
        input_text = self.dataset[idx:idx + self.max_len - 1]
        target_text = self.dataset[idx + 1:idx + self.max_len]

        input_one_hot = torch.Tensor([[0] * len(self.char_idx) for _ in range(self.max_len - 1)])
        for idx, char in enumerate(input_text):
            input_one_hot[idx][self.char_idx[char]] = 1

        return input_one_hot, torch.Tensor([self.char_idx[x] for x in target_text]).type(torch.LongTensor)

    def __len__(self):
        return len(self.dataset) - self.max_len