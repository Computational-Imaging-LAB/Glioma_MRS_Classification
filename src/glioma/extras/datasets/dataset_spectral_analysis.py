import torch
from torch.utils.data import Dataset

class DatasetForSpectralAnalysis(Dataset):
    """Custom dataset for handling spectral data.

    Args:
        X (numpy array): Spectral data.
        age (numpy array): Age information.
        y (numpy array): Response labels.
    """
    def __init__(self, X, age, y):
        self.spectral_data = torch.tensor(X, dtype=torch.float32)
        self.age = torch.tensor(age, dtype=torch.float32)
        self.response = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.spectral_data)

    def __getitem__(self, idx):
        spectrum = self.spectral_data[idx, :].squeeze()
        age = self.age[idx, :].squeeze()
        response = self.response[idx].squeeze()

        # You might want to return age, spectrum, response as a tuple
        return age, spectrum.unsqueeze(0), response
    