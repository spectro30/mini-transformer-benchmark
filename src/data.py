import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

MAX_LEN = 20
VOCAB_SIZE = 5
PAD_ID = 0

token_cols = [f"token_{i:02d}" for i in range(1, MAX_LEN + 1)]
mask_cols = [f"mask_{i:02d}" for i in range(1, MAX_LEN + 1)]


class SequenceDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.tokens = torch.tensor(df[token_cols].values, dtype=torch.long)
        self.masks = torch.tensor(df[mask_cols].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "tokens": self.tokens[idx],
            "mask": self.masks[idx],
            "label": self.labels[idx]
        }


def get_dataloaders(train_path, val_path, test_path, batch_size=32):
    train = SequenceDataset(train_path)
    val = SequenceDataset(val_path)
    test = SequenceDataset(test_path)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
