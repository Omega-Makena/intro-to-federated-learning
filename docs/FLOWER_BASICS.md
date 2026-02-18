# Flower Framework Basics

This guide introduces you to the Flower (flwr) framework for federated learning.

## What is Flower?

Flower (Federated Learning with Flower) is an open-source framework that makes it easy to build federated learning systems. It's designed to be:

- **Simple**: Easy-to-use API for quick prototyping
- **Flexible**: Customize every aspect of federated learning
- **Framework-agnostic**: Works with PyTorch, TensorFlow, JAX, scikit-learn, and more
- **Production-ready**: Scales from simulation to real-world deployment

## Core Concepts

### 1. Client

A client represents a participant in federated learning. It holds local data and performs local training.

**Key Methods:**
- `get_parameters()`: Returns current model parameters
- `set_parameters()`: Sets model parameters from server
- `fit()`: Trains the model on local data
- `evaluate()`: Evaluates the model on local data

**Example Client Structure:**
```python
import flwr as fl

class MyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def get_parameters(self, config):
        """Return current model parameters as a list of numpy arrays."""
        return [param.cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on local data."""
        self.set_parameters(parameters)
        # Training logic here
        train_model(self.model, self.train_loader, epochs=config["epochs"])
        return self.get_parameters(config={}), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        self.set_parameters(parameters)
        # Evaluation logic here
        loss, accuracy = evaluate_model(self.model, self.test_loader)
        return loss, len(self.test_loader), {"accuracy": accuracy}
```

### 2. Server

The server coordinates the federated learning process by:
- Selecting clients for each round
- Distributing the global model
- Receiving model updates
- Aggregating updates into a new global model

**Starting a Server:**
```python
import flwr as fl

# Start server with FedAvg strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=3,  # Minimum clients for training
        min_evaluate_clients=3,  # Minimum clients for evaluation
        min_available_clients=3,  # Minimum clients that must connect
    )
)
```

### 3. Strategy

A strategy defines how the server aggregates client updates. Flower provides several built-in strategies:

#### FedAvg (Federated Averaging)
The most common strategy. Averages model weights from clients, typically weighted by dataset size.

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Sample 50% of clients for training
    fraction_evaluate=0.5,  # Sample 50% of clients for evaluation
    min_fit_clients=10,  # Minimum clients for training
    min_evaluate_clients=5,  # Minimum clients for evaluation
    min_available_clients=10,  # Wait for 10 clients before starting
)
```

#### FedProx
Extension of FedAvg with a proximal term to handle heterogeneous data.

```python
strategy = fl.server.strategy.FedProx(
    proximal_mu=0.1,  # Proximal term coefficient
    # ... other parameters like FedAvg
)
```

#### Custom Strategy
You can create custom strategies by extending `fl.server.strategy.Strategy`:

```python
class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """Custom aggregation logic."""
        # Your custom logic here
        return super().aggregate_fit(rnd, results, failures)
```

## Flower Workflow

### Standard Workflow

```
1. Server starts and waits for clients
   │
   ├─> 2. Clients connect to server
   │
   ├─> 3. Server sends global model to selected clients
   │
   ├─> 4. Clients train model locally (fit)
   │
   ├─> 5. Clients send updated parameters to server
   │
   ├─> 6. Server aggregates parameters
   │
   └─> 7. Repeat steps 3-6 for N rounds
```

### Communication Protocol

**Round N:**
```
Server                          Client 1          Client 2          Client 3
  │                                │                 │                 │
  ├─ fit_config() ────────────────>│                 │                 │
  │                                │                 │                 │
  ├─ parameters ──────────────────>│                 │                 │
  │                                │                 │                 │
  │                             fit(params)       fit(params)      fit(params)
  │                                │                 │                 │
  │<─ updated_params ──────────────┤                 │                 │
  │<─ updated_params ──────────────┼─────────────────┤                 │
  │<─ updated_params ──────────────┼─────────────────┼─────────────────┤
  │                                │                 │                 │
  aggregate()                      │                 │                 │
  │                                │                 │                 │
```

## NumPyClient vs. Client

Flower provides two client interfaces:

### NumPyClient (Recommended for Most Cases)
- Works with NumPy arrays
- Framework-agnostic serialization
- Easier to use with PyTorch, TensorFlow, etc.

```python
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [array1, array2, ...]  # NumPy arrays
    
    def fit(self, parameters, config):
        # parameters is list of NumPy arrays
        return updated_parameters, num_examples, metrics
```

### Client (Advanced)
- Works with Flower's internal parameter representation
- More control over serialization
- Use for advanced scenarios

```python
class MyClient(fl.client.Client):
    def get_parameters(self, ins):
        return ParametersRes(parameters=Parameters(...))
    
    def fit(self, ins):
        return FitRes(parameters=Parameters(...), num_examples=100)
```

## Simulation Mode

Flower provides a simulation mode for testing federated learning on a single machine:

```python
import flwr as fl
from flwr.simulation import start_simulation

def client_fn(cid: str):
    """Create a client with given ID."""
    return MyClient(cid).to_client()

# Run simulation with 10 clients
start_simulation(
    client_fn=client_fn,
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fl.server.strategy.FedAvg(),
)
```

## Configuration

### Server Configuration

```python
config = fl.server.ServerConfig(
    num_rounds=10,  # Number of federated learning rounds
    round_timeout=None,  # Timeout per round (None = no timeout)
)
```

### Client Configuration

Passed to clients via the `config` parameter in `fit()` and `evaluate()`:

```python
# In strategy
def configure_fit(self, server_round, parameters, client_manager):
    config = {
        "server_round": server_round,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    return [(client, FitIns(parameters, config)) for client in clients]
```

## Metrics and Logging

### Client-Side Metrics

Return metrics from `fit()` and `evaluate()`:

```python
def fit(self, parameters, config):
    # ... training ...
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num_examples": len(dataset),
    }
    return parameters, num_examples, metrics

def evaluate(self, parameters, config):
    # ... evaluation ...
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    return loss, num_examples, metrics
```

### Server-Side Aggregation

Aggregate metrics across clients:

```python
class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # results is a list of (ClientProxy, FitRes) tuples
        
        # Aggregate metrics
        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        print(f"Round {server_round} average accuracy: {avg_accuracy}")
        
        return super().aggregate_fit(server_round, results, failures)
```

## Error Handling

### Client Failures

The server strategy receives both results and failures:

```python
def aggregate_fit(self, rnd, results, failures):
    if failures:
        print(f"Round {rnd}: {len(failures)} clients failed")
    
    # Continue with successful results
    if len(results) >= self.min_fit_clients:
        return super().aggregate_fit(rnd, results, failures)
    else:
        return None, {}  # Not enough clients
```

### Timeouts

Set timeouts to handle slow clients:

```python
config = fl.server.ServerConfig(
    num_rounds=10,
    round_timeout=600.0,  # 10 minutes per round
)
```

## Best Practices

### 1. Start Simple
Begin with `NumPyClient` and `FedAvg` before customizing.

### 2. Test with Simulation
Use simulation mode to test before deploying to real clients.

### 3. Monitor Metrics
Log and monitor training metrics to understand convergence.

### 4. Handle Heterogeneity
- Use `min_fit_clients` to ensure enough clients participate
- Consider `FedProx` for non-IID data
- Implement client sampling strategies

### 5. Security Considerations
- Use secure communication (TLS/SSL in production)
- Consider differential privacy
- Implement secure aggregation for sensitive applications

### 6. Resource Management
- Set appropriate timeouts
- Handle client failures gracefully
- Consider client resource constraints (compute, memory, bandwidth)

## Common Patterns

### Pattern 1: Different Local Epochs
```python
def configure_fit(self, server_round, parameters, client_manager):
    # More epochs in early rounds
    epochs = 10 if server_round <= 5 else 5
    config = {"local_epochs": epochs}
    return [(client, FitIns(parameters, config)) for client in clients]
```

### Pattern 2: Learning Rate Scheduling
```python
def configure_fit(self, server_round, parameters, client_manager):
    # Decay learning rate over rounds
    lr = 0.01 * (0.9 ** server_round)
    config = {"learning_rate": lr}
    return [(client, FitIns(parameters, config)) for client in clients]
```

### Pattern 3: Client Selection
```python
def configure_fit(self, server_round, parameters, client_manager):
    # Sample different number of clients based on round
    sample_size = max(2, int(len(client_manager.all()) * 0.5))
    clients = client_manager.sample(num_clients=sample_size)
    return [(client, FitIns(parameters, {})) for client in clients]
```

## Troubleshooting

### Clients Not Connecting
- Check server address and port
- Ensure firewall allows connections
- Verify clients are started after server

### Slow Training
- Reduce `min_fit_clients` for testing
- Increase timeout values
- Check client computational resources
- Profile training code for bottlenecks

### Poor Model Performance
- Check data distribution (IID vs. non-IID)
- Increase number of local epochs
- Adjust learning rate
- Increase number of communication rounds
- Try different aggregation strategies (FedProx, FedOpt)

### Memory Issues
- Reduce batch size in clients
- Use gradient checkpointing
- Implement model compression
- Stream data instead of loading all at once

## Next Steps

Now that you understand Flower basics, you're ready to:
1. Implement your first client
2. Set up a federated learning server
3. Train a model using federated learning
4. Experiment with different strategies and configurations

Check out the credit fraud detection example in this repository to see these concepts in action!

## References

- [Flower Documentation](https://flower.dev/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [Flower API Reference](https://flower.dev/docs/framework/ref-api.html)
- [Federated Learning Tutorial](https://flower.dev/docs/tutorial-series-what-is-federated-learning.html)
