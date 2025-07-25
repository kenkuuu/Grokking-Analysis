import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from typing import Tuple

def get_data_loaders(
    data_dir: str,
    n_train: int,
    batch_size: int,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns train and test data loaders for MNIST.

    Args:
        data_dir (str): The directory to store or load the MNIST data.
        n_train (int): The number of samples for the training set.
        batch_size (int): The batch size for the data loaders.
        seed (int): The random seed for reproducible data splitting.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple with train and test data loaders.
    """
    # Standard normalization for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Load the full training dataset
    full_train_dataset: Dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )

    # Create a reproducible generator for splitting
    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = random_split(
        full_train_dataset, [n_train, len(full_train_dataset) - n_train],
        generator=generator
    )

    # Load the test dataset
    test_dataset: Dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader