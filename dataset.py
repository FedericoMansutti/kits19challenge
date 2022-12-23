from torch.utils.data import Dataset
import torch

class KitsDataset(Dataset):
    def __init__(self, volume, segmentation):
        self.volume = torch.from_numpy(volume.get_fdata())
        self.segmentation = torch.from_numpy(segmentation.get_fdata())

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        return self.volume[idx], self.segmentation[idx]
