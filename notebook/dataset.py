import os
import torch
from torch.utils.data import Dataset

class MusicTokenDataset(Dataset):
    def __init__(
        self,
        token_dir,
        block_size=2048,
        stride=256
    ):
        self.block_size = block_size
        self.stride = stride
        self.samples = []

        files = sorted([
            os.path.join(token_dir, f)
            for f in os.listdir(token_dir)
            if f.endswith(".pt")
        ])

        for f in files:
            data = torch.load(f)

            chunks = data["tokens"]         # list of [4, 750]
            tokens = torch.cat(chunks, dim=1)  # [4, T]
            tokens = tokens.permute(1, 0)      # [T, 4]

            T = tokens.size(0)

            # 加 stride
            for i in range(0, T - block_size - 1, stride):
                x = tokens[i:i+block_size]
                y = tokens[i+1:i+block_size+1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]