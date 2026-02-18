"""
Federated Learning Server using Flower

This module implements the server-side of federated learning
for credit card fraud detection. The server:
1. Coordinates communication rounds
2. Distributes the global model to clients
3. Aggregates model updates from clients
4. Evaluates the global model
"""

import argparse
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Dict containing aggregated metrics
    """
    # Calculate total examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        # Aggregate each metric
        for key in metric_keys:
            weighted_sum = sum(
                num_examples * m[key] 
                for num_examples, m in metrics 
                if key in m
            )
            aggregated[key] = weighted_sum / total_examples
    
    return aggregated


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate training metrics from clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
    
    Returns:
        Dict containing aggregated training metrics
    """
    aggregated = weighted_average(metrics)
    
    # Print training metrics
    if aggregated:
        print("\n" + "="*60)
        print("Training Metrics (Aggregated)")
        print("="*60)
        for key, value in aggregated.items():
            print(f"  {key}: {value:.4f}")
        print("="*60 + "\n")
    
    return aggregated


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
    
    Returns:
        Dict containing aggregated evaluation metrics
    """
    aggregated = weighted_average(metrics)
    
    # Print evaluation metrics
    if aggregated:
        print("\n" + "="*60)
        print("Evaluation Metrics (Aggregated)")
        print("="*60)
        for key, value in aggregated.items():
            print(f"  {key}: {value:.4f}")
        print("="*60 + "\n")
    
    return aggregated


def create_strategy(
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    local_epochs: int = 5
) -> FedAvg:
    """
    Create a FedAvg strategy for federated learning.
    
    Args:
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of clients that must be available
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        local_epochs: Number of local training epochs
    
    Returns:
        FedAvg strategy
    """
    
    def fit_config(server_round: int) -> Dict:
        """
        Configure training for each round.
        
        Args:
            server_round: Current round number
        
        Returns:
            Dict with training configuration
        """
        config = {
            "server_round": server_round,
            "local_epochs": local_epochs,
        }
        return config
    
    # Create strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
    )
    
    return strategy


def main():
    """Main function to start the federated learning server."""
    parser = argparse.ArgumentParser(description="Flower Server for Fraud Detection")
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=3,
        help="Minimum number of clients required"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
        help="Number of local training epochs per round"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=1.0,
        help="Fraction of clients to use for training (0.0 to 1.0)"
    )
    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=1.0,
        help="Fraction of clients to use for evaluation (0.0 to 1.0)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Federated Learning Server for Credit Fraud Detection")
    print("="*60)
    print(f"Server address: {args.server_address}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Minimum clients: {args.min_clients}")
    print(f"Local epochs per round: {args.local_epochs}")
    print(f"Fraction fit: {args.fraction_fit}")
    print(f"Fraction evaluate: {args.fraction_evaluate}")
    print("="*60 + "\n")
    
    # Create strategy
    strategy = create_strategy(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        local_epochs=args.local_epochs
    )
    
    print(f"Waiting for {args.min_clients} clients to connect...\n")
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*60)
    print("Federated Learning Completed!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Total rounds: {args.num_rounds}")
    print(f"  Clients participated: {args.min_clients}")
    print("\nThe global model has been trained using federated learning.")
    print("All client data remained local throughout the process.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
