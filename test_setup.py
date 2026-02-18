#!/usr/bin/env python
"""
Simple test to verify federated learning setup works correctly.
"""

import sys
import time
import subprocess
import os

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")
    try:
        import torch
        import flwr
        import numpy
        import pandas
        import sklearn
        print("✓ All imports successful")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - Flower: {flwr.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model():
    """Test model creation."""
    print("\nTesting model...")
    try:
        from src.model import FraudDetectionModel
        model = FraudDetectionModel()
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_data_loader():
    """Test data loading."""
    print("\nTesting data loader...")
    try:
        from src.data_loader import create_synthetic_data, split_data_federated
        X, y = create_synthetic_data(n_samples=1000)
        client_data = split_data_federated(X, y, num_clients=3)
        print(f"✓ Data loader works")
        print(f"  - Created data for {len(client_data)} clients")
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_client_creation():
    """Test client creation."""
    print("\nTesting client creation...")
    try:
        import torch
        from src.client import create_client
        from src.data_loader import create_synthetic_data
        import numpy as np
        
        # Create small dataset
        X, y = create_synthetic_data(n_samples=100)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        device = torch.device("cpu")
        client = create_client(0, X_train, y_train, X_test, y_test, device)
        print(f"✓ Client created successfully")
        return True
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Federated Learning Setup Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_model,
        test_data_loader,
        test_client_creation,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Federated learning setup is ready.")
        print("\nNext steps:")
        print("  1. Run: ./run_federated_learning.sh")
        print("  2. Or manually start server and clients")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
