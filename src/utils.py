"""
Utility functions for federated learning
"""

import os
import json
import time
from datetime import datetime
import torch


def save_model(model, filepath):
    """
    Save a PyTorch model to disk.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """
    Load a PyTorch model from disk.
    
    Args:
        model: PyTorch model (architecture must match saved model)
        filepath: Path to the saved model
    
    Returns:
        model: Model with loaded weights
    """
    # Use weights_only=True for security (PyTorch >= 2.6.0)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    print(f"Model loaded from {filepath}")
    return model


def save_metrics(metrics, filepath):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath):
    """
    Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
    
    Returns:
        dict: Metrics dictionary
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_str(self):
        """Get elapsed time as formatted string."""
        elapsed = self.elapsed()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_section(title, width=60):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model):
    """
    Print information about a model.
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test timer
    print("\n1. Testing Timer...")
    timer = Timer()
    timer.start()
    time.sleep(1)
    print(f"   Elapsed: {timer.elapsed_str()}")
    
    # Test section printer
    print_section("Test Section")
    
    # Test timestamp
    print(f"Current timestamp: {get_timestamp()}")
    
    print("\n Utility functions test completed!")
