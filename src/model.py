"""
PyTorch Model for Credit Card Fraud Detection

This module defines a neural network for binary classification
of credit card transactions (fraudulent or legitimate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudDetectionModel(nn.Module):
    """
    Neural Network for Credit Card Fraud Detection.
    
    Architecture:
    - Input layer: 30 features (Time, Amount, V1-V28)
    - Hidden layers: Multiple fully connected layers with ReLU and Dropout
    - Output layer: 2 classes (fraud or legitimate)
    
    The model uses:
    - Batch normalization for stable training
    - Dropout for regularization
    - ReLU activation for non-linearity
    """
    
    def __init__(self, input_dim=30, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        """
        Initialize the fraud detection model.
        
        Args:
            input_dim (int): Number of input features (default: 30)
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability for regularization
        """
        super(FraudDetectionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with batch norm and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (2 classes: legitimate and fraud)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, 2)
        """
        return self.network(x)
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Predicted class labels (0 or 1)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x):
        """
        Get prediction probabilities.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Probability of each class
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def get_model_parameters(model):
    """
    Extract model parameters as a list of numpy arrays.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        list: List of numpy arrays containing model parameters
    """
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_parameters(model, parameters):
    """
    Set model parameters from a list of numpy arrays.
    
    Args:
        model (nn.Module): PyTorch model
        parameters (list): List of numpy arrays containing parameters
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_model(model, train_loader, criterion, optimizer, device, epochs=1):
    """
    Train the model for specified number of epochs.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or CUDA)
        epochs (int): Number of training epochs
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()
        
        total_loss += epoch_loss / len(train_loader)
        correct += epoch_correct
        total += epoch_total
    
    avg_loss = total_loss / epochs
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device: Device to evaluate on (CPU or CUDA)
    
    Returns:
        tuple: (loss, accuracy, precision, recall, f1_score)
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # For precision, recall, F1
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Calculate confusion matrix components
            true_positives += ((predicted == 1) & (target == 1)).sum().item()
            false_positives += ((predicted == 1) & (target == 0)).sum().item()
            false_negatives += ((predicted == 0) & (target == 1)).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return avg_loss, accuracy, precision, recall, f1_score


if __name__ == "__main__":
    # Test the model
    print("Testing FraudDetectionModel...")
    
    # Create model
    model = FraudDetectionModel(input_dim=30, hidden_dims=[64, 32, 16])
    print(f"\nModel Architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 30)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Test probability prediction
    probabilities = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities:\n{probabilities[:5]}")
    
    print("\n✓ Model test completed successfully!")
