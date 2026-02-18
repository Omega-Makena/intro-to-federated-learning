"""
Federated Learning Client using Flower

This module implements a Flower client for credit card fraud detection.
Each client trains a model on its local data and communicates only
model updates to the server.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from collections import OrderedDict

# Add src directory to path if running from project root
if os.path.basename(os.getcwd()) == 'intro-to-federated-learning':
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from model import FraudDetectionModel, train_model, evaluate_model
from data_loader import (
    create_synthetic_data,
    split_data_federated,
    get_dataloader,
    get_class_weights,
    print_data_statistics
)


class FraudDetectionClient(fl.client.NumPyClient):
    """
    Flower client for federated learning of fraud detection.
    
    This client:
    1. Receives the global model from server
    2. Trains on local data
    3. Sends model updates back to server
    """
    
    def __init__(self, client_id, train_loader, test_loader, class_weights, device):
        """
        Initialize the client.
        
        Args:
            client_id (int): Unique identifier for this client
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            class_weights (torch.Tensor): Weights for handling class imbalance
            device: Device to train on (CPU or CUDA)
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Initialize model
        self.model = FraudDetectionModel(
            input_dim=30,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.3
        ).to(device)
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"Client {client_id} initialized with {len(train_loader.dataset)} training samples")
    
    def get_parameters(self, config):
        """
        Get model parameters as NumPy arrays.
        
        Args:
            config (dict): Configuration dictionary (not used here)
        
        Returns:
            list: List of NumPy arrays containing model parameters
        """
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from NumPy arrays.
        
        Args:
            parameters (list): List of NumPy arrays containing model parameters
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model on local data.
        
        Args:
            parameters (list): Current global model parameters
            config (dict): Training configuration from server
        
        Returns:
            tuple: (updated_parameters, num_examples, metrics)
        """
        print(f"\n[Client {self.client_id}] Starting local training...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("local_epochs", 1)
        
        # Train model
        train_loss, train_accuracy = train_model(
            self.model,
            self.train_loader,
            self.criterion,
            self.optimizer,
            self.device,
            epochs=epochs
        )
        
        print(f"[Client {self.client_id}] Training completed:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_accuracy:.4f}")
        
        # Return updated model parameters and metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data.
        
        Args:
            parameters (list): Model parameters to evaluate
            config (dict): Evaluation configuration from server
        
        Returns:
            tuple: (loss, num_examples, metrics)
        """
        print(f"\n[Client {self.client_id}] Evaluating model...")
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        test_loss, accuracy, precision, recall, f1_score = evaluate_model(
            self.model,
            self.test_loader,
            self.criterion,
            self.device
        )
        
        print(f"[Client {self.client_id}] Evaluation results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        
        # Return metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
        
        return test_loss, len(self.test_loader.dataset), metrics


def create_client(client_id, X_train, y_train, X_test, y_test, device):
    """
    Create a Flower client with data.
    
    Args:
        client_id (int): Client identifier
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        device: Device to use for training
    
    Returns:
        FraudDetectionClient: Initialized client
    """
    # Create data loaders
    train_loader = get_dataloader(X_train, y_train, batch_size=32, use_sampler=True)
    test_loader = get_dataloader(X_test, y_test, batch_size=32, shuffle=False)
    
    # Calculate class weights
    class_weights = get_class_weights(y_train)
    
    # Create client
    client = FraudDetectionClient(
        client_id=client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        device=device
    )
    
    return client


def main():
    """Main function to start a federated learning client."""
    parser = argparse.ArgumentParser(description="Flower Client for Fraud Detection")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="Client ID (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Total number of clients in the simulation"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="iid",
        choices=["iid", "non-iid"],
        help="Type of data split (IID or Non-IID)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Starting Federated Learning Client {args.client_id}")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\nGenerating synthetic credit card data...")
    X, y = create_synthetic_data(n_samples=10000, fraud_ratio=0.002)
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Split data among clients
    print(f"\nCreating {args.data_split.upper()} data split for {args.num_clients} clients...")
    client_data = split_data_federated(
        X_train, y_train,
        num_clients=args.num_clients,
        iid=(args.data_split == "iid")
    )
    
    # Get this client's data
    if args.client_id >= len(client_data):
        raise ValueError(f"Client ID {args.client_id} out of range (0-{len(client_data)-1})")
    
    X_train_client, y_train_client = client_data[args.client_id]
    
    print(f"\nClient {args.client_id} data:")
    print(f"  Training samples: {len(X_train_client):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Fraud rate (train): {sum(y_train_client)/len(y_train_client)*100:.3f}%")
    
    # Create client
    print("\nInitializing client...")
    client = create_client(
        client_id=args.client_id,
        X_train=X_train_client,
        y_train=y_train_client,
        X_test=X_test,
        y_test=y_test,
        device=device
    )
    
    # Start Flower client
    print(f"\nConnecting to server at {args.server_address}...")
    print("Waiting for training rounds to begin...\n")
    
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )
    
    print("\n" + "="*60)
    print(f"Client {args.client_id} finished!")
    print("="*60)


if __name__ == "__main__":
    main()
