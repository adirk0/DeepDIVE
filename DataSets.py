import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# using the RepeatedOneHotDataset DataSet
def get_DataLoader(input_dim, batch_size, epoch_size, device="cpu", shuffle=True):
    repeat_factor = 1+(epoch_size // input_dim)

    # Generate data (one-hot vectors for each class)
    data = torch.eye(input_dim)
    data = data.to(device)

    # Create the dataset with repeating
    dataset = RepeatedOneHotDataset(data, repeat_factor=repeat_factor)
    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


# using the SampledOneHotDataset DataSet
def get_SampledDataLoader(input_dim, probabilities, batch_size, epoch_size, device="cpu", shuffle=True):
    repeat_factor = epoch_size // input_dim

    # Generate data (one-hot vectors for each class)
    data = torch.eye(input_dim)
    data = data.to(device)

    # Create the dataset with repeating
    dataset = SampledOneHotDataset(data, repeat_factor=repeat_factor, probabilities=probabilities)
    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


class RepeatedOneHotDataset(Dataset):
    def __init__(self, data, repeat_factor=1):
        self.data = data
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.data) * self.repeat_factor

    def __getitem__(self, idx):
        idx = idx % len(self.data)  # Ensure idx is within the bounds of the data
        return self.data[idx]


class SampledOneHotDataset(Dataset):
    def __init__(self, data, repeat_factor=1, probabilities=None):
        self.data = data
        self.repeat_factor = repeat_factor

        # Ensure probabilities are provided correctly
        if probabilities is None:
            probabilities = torch.ones(len(data)) / len(data)  # Default: uniform distribution
        else:
            assert len(probabilities) == len(data), "Length of probabilities must match the number of data points"
            probabilities = torch.tensor(probabilities, dtype=torch.float32)
            probabilities /= probabilities.sum()  # Normalize probabilities to sum to 1

        # Calculate the cumulative distribution for efficient indexing
        self.cumulative_probs = torch.cumsum(probabilities, dim=0)

    def __len__(self):
        return len(self.data) * self.repeat_factor

    def __getitem__(self, idx):
        # Map idx to a corresponding data point based on cumulative probabilities
        idx = (idx % (len(self.data) * self.repeat_factor)) / (len(self.data) * self.repeat_factor)  # Normalize to the range [0, 1)

        # Find the corresponding data point using cumulative probabilities
        data_idx = torch.searchsorted(self.cumulative_probs, idx, right=True)
        return self.data[data_idx]