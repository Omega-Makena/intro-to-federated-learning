"""
Data Loading and Preprocessing for Credit Card Fraud Detection

This module handles:
- Loading credit card transaction data
- Preprocessing and normalization
- Creating federated data splits (simulating multiple clients)
- Handling class imbalance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class CreditCardDataset(Dataset):
    """PyTorch Dataset for credit card transactions."""
    
    def __init__(self, features, labels):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Label vector
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """
    Load and preprocess credit card fraud detection data.
    
    Expected CSV format:
    - Columns: Time, V1-V28, Amount, Class
    - Class: 0 (legitimate) or 1 (fraud)
    
    Args:
        filepath (str): Path to the CSV file
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_synthetic_data(n_samples=10000, n_features=30, fraud_ratio=0.002, random_state=42):
    """
    Create synthetic credit card transaction data for demonstration.
    
    This is useful when the real dataset is not available.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        fraud_ratio (float): Ratio of fraudulent transactions
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) features and labels
    """
    np.random.seed(random_state)
    
    # Number of fraud and legitimate samples
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Generate legitimate transactions (centered around 0)
    X_legit = np.random.randn(n_legit, n_features)
    y_legit = np.zeros(n_legit, dtype=np.int64)
    
    # Generate fraudulent transactions (with different distribution)
    X_fraud = np.random.randn(n_fraud, n_features) * 2 + 1  # Different mean and variance
    y_fraud = np.ones(n_fraud, dtype=np.int64)
    
    # Combine and shuffle
    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([y_legit, y_fraud])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def split_data_federated(X, y, num_clients=3, iid=True, random_state=42):
    """
    Split data among multiple clients for federated learning simulation.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        num_clients (int): Number of clients to split data among
        iid (bool): If True, split data IID; if False, create non-IID split
        random_state (int): Random seed
    
    Returns:
        list: List of tuples (X_client, y_client) for each client
    """
    np.random.seed(random_state)
    n_samples = len(X)
    
    if iid:
        # IID split: randomly distribute data among clients
        indices = np.random.permutation(n_samples)
        split_indices = np.array_split(indices, num_clients)
        
        client_data = []
        for idx in split_indices:
            client_data.append((X[idx], y[idx]))
    
    else:
        # Non-IID split: each client has different class distribution
        # Sort by label and split into chunks
        sorted_indices = np.argsort(y)
        
        # Create chunks where each client gets different portions of classes
        client_data = []
        chunk_size = n_samples // num_clients
        
        for i in range(num_clients):
            # Each client gets a different window of sorted data
            start_idx = (i * chunk_size) % n_samples
            end_idx = ((i + 1) * chunk_size) % n_samples
            
            if start_idx < end_idx:
                client_indices = sorted_indices[start_idx:end_idx]
            else:
                # Wrap around
                client_indices = np.concatenate([
                    sorted_indices[start_idx:],
                    sorted_indices[:end_idx]
                ])
            
            # Add some randomness
            np.random.shuffle(client_indices)
            client_data.append((X[client_indices], y[client_indices]))
    
    return client_data


def create_balanced_sampler(labels):
    """
    Create a weighted sampler to handle class imbalance.
    
    Args:
        labels (torch.Tensor or np.ndarray): Class labels
    
    Returns:
        WeightedRandomSampler: Sampler for balanced batches
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Calculate class weights
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler


def get_dataloader(X, y, batch_size=32, shuffle=True, use_sampler=False):
    """
    Create a PyTorch DataLoader.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        use_sampler (bool): Whether to use balanced sampler
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = CreditCardDataset(X, y)
    
    if use_sampler:
        sampler = create_balanced_sampler(y)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_class_weights(labels):
    """
    Calculate class weights for handling imbalanced data in loss function.
    
    Args:
        labels (np.ndarray or torch.Tensor): Class labels
    
    Returns:
        torch.Tensor: Class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Count samples per class
    class_counts = np.bincount(labels)
    
    # Calculate weights (inverse of frequency)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(class_weights)


def print_data_statistics(client_data):
    """
    Print statistics about federated data distribution.
    
    Args:
        client_data (list): List of (X, y) tuples for each client
    """
    print("\n" + "="*60)
    print("Federated Data Distribution")
    print("="*60)
    
    for i, (X_client, y_client) in enumerate(client_data):
        n_samples = len(y_client)
        n_fraud = np.sum(y_client == 1)
        n_legit = np.sum(y_client == 0)
        fraud_rate = n_fraud / n_samples * 100
        
        print(f"\nClient {i}:")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Legitimate: {n_legit:,} ({100-fraud_rate:.2f}%)")
        print(f"  Fraudulent: {n_fraud:,} ({fraud_rate:.2f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Testing data loading and preprocessing...")
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    X, y = create_synthetic_data(n_samples=10000, fraud_ratio=0.002)
    print(f"   Generated {len(X):,} samples with {X.shape[1]} features")
    print(f"   Fraud ratio: {np.mean(y) * 100:.3f}%")
    
    # Split data
    print("\n2. Splitting data (train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Create federated split
    print("\n3. Creating federated data split (3 clients, IID)...")
    client_data_iid = split_data_federated(X_train, y_train, num_clients=3, iid=True)
    print_data_statistics(client_data_iid)
    
    print("\n4. Creating federated data split (3 clients, Non-IID)...")
    client_data_non_iid = split_data_federated(X_train, y_train, num_clients=3, iid=False)
    print_data_statistics(client_data_non_iid)
    
    # Create DataLoader
    print("\n5. Creating DataLoader...")
    X_client, y_client = client_data_iid[0]
    train_loader = get_dataloader(X_client, y_client, batch_size=32, use_sampler=True)
    print(f"   Created DataLoader with {len(train_loader)} batches")
    
    # Test batch
    features, labels = next(iter(train_loader))
    print(f"   Batch shape: features={features.shape}, labels={labels.shape}")
    
    # Calculate class weights
    print("\n6. Calculating class weights...")
    class_weights = get_class_weights(y_train)
    print(f"   Class weights: {class_weights}")
    
    print("\n✓ Data loading test completed successfully!")
