<<<<<<< HEAD
# intro-to-federated-learning
=======
# Introduction to Federated Learning with Flower

A comprehensive introduction to Federated Learning using the Flower framework, featuring a practical credit card fraud detection example with PyTorch.

> ** Security Note**: This project uses PyTorch >=2.6.0 to address known security vulnerabilities. See [SECURITY.md](SECURITY.md) for details.
>
> ** Dataset Note**: This project uses synthetic credit card data by default (no download required). See [DATASET.md](DATASET.md) for complete information about the dataset.

## Table of Contents
- [What is Federated Learning?](#what-is-federated-learning)
- [Why Federated Learning?](#why-federated-learning)
- [Key Concepts](#key-concepts)
- [What is Flower?](#what-is-flower)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Credit Fraud Detection Example](#credit-fraud-detection-example)
- [How It Works](#how-it-works)
- [Resources](#resources)

## What is Federated Learning?

Federated Learning (FL) is a machine learning approach that enables training models across multiple decentralized devices or servers holding local data samples, without exchanging the actual data. Instead of bringing data to the model, federated learning brings the model to the data.

### Traditional Machine Learning vs. Federated Learning

**Traditional ML:**
```
[Device 1 Data] ─┐
[Device 2 Data] ─┼─> [Central Server] -> [Train Model]
[Device 3 Data] ─┘
```
All data is collected in one place for training.

**Federated Learning:**
```
[Device 1] -> Local Training -> Model Updates ─┐
[Device 2] -> Local Training -> Model Updates ─┼─> [Central Server] -> Aggregate -> Global Model
[Device 3] -> Local Training -> Model Updates ─┘
```
Data stays on devices; only model updates are shared.

## Why Federated Learning?

### Privacy Preservation
- **Data remains local**: Sensitive data never leaves the device
- **Regulatory compliance**: Helps meet GDPR, HIPAA, and other privacy regulations
- **User trust**: Users maintain control over their data

### Practical Benefits
- **Reduced bandwidth**: Only model updates are transmitted, not raw data
- **Lower latency**: Models can be trained on edge devices
- **Scalability**: Leverage computational power of distributed devices
- **Real-world scenarios**: Healthcare, finance, mobile keyboards, IoT devices

## Key Concepts

### 1. **Clients (Workers)**
Devices or servers that hold local data and perform local training. Each client:
- Trains the model on its local dataset
- Computes model updates (gradients or weights)
- Sends updates to the server

### 2. **Server (Aggregator)**
Central coordinator that:
- Distributes the global model to clients
- Receives model updates from clients
- Aggregates updates using strategies like FedAvg (Federated Averaging)
- Produces a new global model

### 3. **Federated Averaging (FedAvg)**
The most common aggregation algorithm:
1. Server sends global model to selected clients
2. Clients train on local data for several epochs
3. Clients send updated weights back to server
4. Server averages the weights (often weighted by dataset size)
5. Process repeats for multiple rounds

### 4. **Communication Rounds**
One complete cycle of:
- Model distribution
- Local training
- Update aggregation

## What is Flower?

[Flower (flwr)](https://flower.dev/) is a friendly federated learning framework that makes it easy to build federated learning systems. 

### Why Flower?

- **Framework agnostic**: Works with PyTorch, TensorFlow, JAX, and more
- **Easy to use**: Simple API for both research and production
- **Flexible**: Customizable aggregation strategies and client selection
- **Scalable**: From simulation to real-world deployment
- **Well-maintained**: Active community and regular updates

### Core Components

1. **`flwr.server.Server`**: Manages the federated learning process
2. **`flwr.client.Client`**: Defines client behavior and training logic
3. **`flwr.server.strategy`**: Aggregation strategies (FedAvg, FedProx, etc.)
4. **`flwr.simulation`**: Tools for simulating federated learning scenarios

## Project Structure

```
intro-to-federated-learning/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── README.md                      # Data information
├── notebooks/
│   ├── 00_pytorch_basics.ipynb        # PyTorch fundamentals tutorial
│   ├── 01_flower_basics.ipynb        # Introduction to Flower
│   └── 02_federated_learning_demo.ipynb  # Complete FL demo
├── src/
│   ├── client.py                      # Federated learning client
│   ├── server.py                      # Federated learning server
│   ├── model.py                       # PyTorch fraud detection model
│   ├── data_loader.py                 # Data loading and preprocessing
│   └── utils.py                       # Utility functions
└── docs/
    ├── FLOWER_BASICS.md              # Flower library basics
    └── FEDERATED_LEARNING_CONCEPTS.md # FL concepts explained
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Omega-Makena/intro-to-federated-learning.git
cd intro-to-federated-learning
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### 0. Learn PyTorch Basics (Optional but Recommended)
If you're new to PyTorch, start with our interactive tutorial:
```bash
jupyter notebook notebooks/00_pytorch_basics.ipynb
```

This covers:
- Tensors and operations
- Building neural networks with `nn.Module`
- Training loops and optimization
- Model evaluation and persistence
- Connection to federated learning

### 1. Understand Flower Basics
Read the documentation:
```bash
cat docs/FLOWER_BASICS.md
```

Or explore the interactive notebook:
```bash
jupyter notebook notebooks/01_flower_basics.ipynb
```

### 2. Run the Federated Learning Simulation

**Start the server:**
```bash
python src/server.py
```

**In separate terminals, start clients:**
```bash
python src/client.py --client-id 0
python src/client.py --client-id 1
python src/client.py --client-id 2
```

### 3. Monitor Training
Watch as the federated learning process:
- Distributes the model to clients
- Trains locally on each client's data
- Aggregates updates at the server
- Improves the global model over multiple rounds

## Credit Fraud Detection Example

This project demonstrates federated learning with a **credit card fraud detection** use case, which is ideal for FL because:

- **Privacy-sensitive**: Financial transaction data should remain local
- **Distributed data**: Different banks/institutions hold their own data
- **Imbalanced data**: Fraud cases are rare (class imbalance challenge)
- **Real-world relevance**: Practical application of FL

### Dataset

> ** Dataset Used: Synthetic Data (Default)**
> 
> **By default, this project uses synthetic credit card transaction data** that mimics the structure and characteristics of real fraud detection datasets. This allows you to:
> - Run the example immediately without downloading large datasets
> - Learn federated learning concepts quickly
> - Test the implementation in minutes
>
> The synthetic data has the same structure as the real dataset (30 features, ~0.2% fraud rate).

#### Option 1: Synthetic Data (Current Default) 

The implementation automatically generates synthetic credit card transactions with:
- 10,000 transactions (configurable)
- 30 features (Time, V1-V28, Amount)
- ~0.2% fraud rate (realistic imbalance)
- Same structure as real Credit Card Fraud Detection dataset

**No setup required** - just run the code and it works!

#### Option 2: Real Kaggle Dataset (Optional)

You can optionally use the real [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud):
- 284,807 real transactions
- 492 actual frauds (0.172% of all transactions)
- Features: Time, Amount, and 28 anonymized features (V1-V28)
- Binary classification: Fraud (1) or Legitimate (0)

**To use the real dataset:**
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Place it in the `data/` directory
3. Modify `src/client.py` to use `load_and_preprocess_data("data/creditcard.csv")` instead of `create_synthetic_data()`

See [`data/README.md`](data/README.md) for detailed instructions.

### Model Architecture

A PyTorch neural network with:
- Input layer: 30 features
- Hidden layers: Multiple fully connected layers with ReLU activation
- Dropout: For regularization
- Output layer: Binary classification (fraud or not)

### Federated Setup

- **3 Clients**: Simulating 3 different financial institutions
- **Non-IID Data**: Each client has different data distribution
- **Class Imbalance Handling**: Weighted loss function
- **Aggregation**: FedAvg strategy
- **Rounds**: 10-20 communication rounds

## How It Works

### Step-by-Step Process

1. **Data Preparation**
   - Download credit card fraud dataset
   - Split data among multiple clients (simulating different banks)
   - Handle class imbalance

2. **Model Definition**
   - Define PyTorch neural network for fraud detection
   - Implement forward pass and training logic

3. **Client Implementation**
   - Create Flower client class
   - Implement `fit()` for local training
   - Implement `evaluate()` for local evaluation
   - Handle model parameters (get/set)

4. **Server Implementation**
   - Configure Flower server
   - Set aggregation strategy (FedAvg)
   - Define number of rounds and client selection

5. **Training**
   - Server sends initial model to clients
   - Each client trains on local data
   - Clients send updates to server
   - Server aggregates and creates new global model
   - Repeat for multiple rounds

6. **Evaluation**
   - Evaluate global model performance
   - Compare with centralized training
   - Analyze privacy-utility trade-offs

## Resources

### Flower Documentation
- [Official Flower Documentation](https://flower.dev/docs/)
- [Flower GitHub Repository](https://github.com/adap/flower)
- [Flower Tutorials](https://flower.dev/docs/tutorial-series-what-is-federated-learning.html)

### Federated Learning Papers
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (FedAvg paper)
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)

### Additional Learning
- [Federated Learning Blog by Google AI](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Flower team for the excellent federated learning framework
- The machine learning community for advancing federated learning research
- Credit card fraud detection dataset providers
>>>>>>> copilot/add-introduction-to-federated-learning
