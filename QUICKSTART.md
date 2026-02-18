# Quick Start Guide

Get started with federated learning in 5 minutes!

## Step 1: Installation

### Clone the repository
```bash
git clone https://github.com/Omega-Makena/intro-to-federated-learning.git
cd intro-to-federated-learning
```

### Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `flwr` - Flower federated learning framework
- `torch` - PyTorch for neural networks
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Machine learning utilities
- `matplotlib`, `seaborn` - Visualization

## Step 2: Learn the Basics

### PyTorch Tutorial (Optional but Recommended)
If you're new to PyTorch, start here:
```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/00_pytorch_basics.ipynb
```

This interactive tutorial covers:
- Tensors and neural networks
- Training loops and optimization
- How PyTorch connects to federated learning

### Read the documentation
```bash
# Federated Learning concepts
cat docs/FEDERATED_LEARNING_CONCEPTS.md

# Flower framework basics
cat docs/FLOWER_BASICS.md
```

### Explore Jupyter notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in your browser:
# - notebooks/00_pytorch_basics.ipynb (PyTorch fundamentals)
# - notebooks/01_flower_basics.ipynb (Flower framework)
# - notebooks/02_federated_learning_demo.ipynb (Complete FL demo)
```

## Step 3: Run Federated Learning Simulation

### Option A: Using the run script (easiest)

```bash
chmod +x run_federated_learning.sh
./run_federated_learning.sh
```

This automatically:
- Starts the federated learning server
- Launches 3 client instances
- Runs training for 10 rounds
- Shows aggregated results

### Option B: Manual setup (more control)

**Terminal 1 - Start Server:**
```bash
python src/server.py \
    --num-rounds 10 \
    --min-clients 3 \
    --local-epochs 5
```

**Terminal 2 - Start Client 0:**
```bash
python src/client.py \
    --client-id 0 \
    --server-address 127.0.0.1:8080
```

**Terminal 3 - Start Client 1:**
```bash
python src/client.py \
    --client-id 1 \
    --server-address 127.0.0.1:8080
```

**Terminal 4 - Start Client 2:**
```bash
python src/client.py \
    --client-id 2 \
    --server-address 127.0.0.1:8080
```

## Step 4: Understand the Output

### Server Output
```
============================================================
Federated Learning Server for Credit Fraud Detection
============================================================
Server address: 0.0.0.0:8080
Number of rounds: 10
Minimum clients: 3
...

Round 1:
  Training Metrics (Aggregated)
    train_loss: 0.2345
    train_accuracy: 0.9876
...
```

### Client Output
```
Client 0 initialized with 2,667 training samples

[Client 0] Starting local training...
[Client 0] Training completed:
  Loss: 0.2156
  Accuracy: 0.9891
```

## Step 5: Experiment

### Try different configurations

**More rounds:**
```bash
python src/server.py --num-rounds 20
```

**Non-IID data:**
```bash
python src/client.py --client-id 0 --data-split non-iid
```

**Different local epochs:**
```bash
python src/server.py --local-epochs 10
```

## What's Happening Behind the Scenes?

### 1. Data Preparation

> ** Dataset: Synthetic Data (Default)**
>
> The project uses **synthetic credit card transaction data** by default. No dataset download required!

- Synthetic credit card data is automatically generated
- Data is split among 3 clients (simulating 3 banks)
- Each client has ~2,600 transactions
- Class imbalance: ~0.2% fraud rate (realistic!)
- 30 features matching real Credit Card Fraud Detection dataset structure

**Want to use the real Kaggle dataset instead?** See [`data/README.md`](data/README.md) for instructions.

### 2. Federated Training Process

```
Round N:
  1. Server sends global model to clients
  2. Each client trains on LOCAL data (privacy!)
  3. Clients send model UPDATES (not data!)
  4. Server aggregates updates (FedAvg)
  5. Global model improves
```

### 3. Privacy Preservation
 **Client data never leaves the device**
 **Only model weights are shared**
 **Server never sees raw transactions**

## Common Issues & Solutions

### Issue: "No module named 'flwr'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Connection refused"
**Solution:** Start server first, then clients
```bash
# Terminal 1: Start server first
python src/server.py

# Then start clients in other terminals
```

### Issue: "Not enough clients"
**Solution:** Start all required clients (default: 3)
```bash
# Server waits for 3 clients
python src/client.py --client-id 0  # Terminal 2
python src/client.py --client-id 1  # Terminal 3
python src/client.py --client-id 2  # Terminal 4
```

## Next Steps

### Beginner
-  Read `README.md` for comprehensive overview
-  Complete both Jupyter notebooks
-  Run simulation with default settings
-  Try different number of rounds (5, 10, 20)

### Intermediate
-  Download real credit card fraud dataset from Kaggle
-  Modify model architecture in `src/model.py`
-  Experiment with IID vs Non-IID data splits
-  Adjust learning rates and batch sizes

### Advanced
-  Implement custom aggregation strategy
-  Add differential privacy
-  Deploy to multiple machines
-  Scale to more clients (10+)
-  Implement secure aggregation

## Project Structure
```
intro-to-federated-learning/
├── README.md                  # Main documentation
├── requirements.txt           # Python dependencies
├── run_federated_learning.sh # Quick start script
├── docs/
│   ├── FLOWER_BASICS.md      # Flower framework guide
│   └── FEDERATED_LEARNING_CONCEPTS.md  # FL concepts
├── notebooks/
│   ├── 01_flower_basics.ipynb          # Interactive tutorial
│   └── 02_federated_learning_demo.ipynb # Complete demo
├── src/
│   ├── server.py             # FL server
│   ├── client.py             # FL client
│   ├── model.py              # Neural network
│   ├── data_loader.py        # Data handling
│   └── utils.py              # Utilities
└── data/
    └── README.md             # Dataset information
```

## Key Concepts Recap

### Federated Learning
- **Decentralized**: Data stays on clients
- **Privacy-preserving**: Only model updates shared
- **Collaborative**: Multiple parties learn together

### Flower Framework
- **Easy to use**: Simple API (fit, evaluate)
- **Flexible**: Custom strategies and configurations
- **Scalable**: Simulation to production

### Credit Fraud Detection
- **Real-world use case**: Financial transactions
- **Class imbalance**: Rare fraud events
- **Privacy-critical**: Sensitive financial data

## Resources

- **Flower**: https://flower.dev/
- **Dataset**: https://www.kaggle.com/mlg-ulb/creditcardfraud
- **PyTorch**: https://pytorch.org/
- **FL Paper**: https://arxiv.org/abs/1602.05629

## Getting Help

- Read the documentation in `docs/`
- Check examples in `notebooks/`
- Review source code in `src/`
- Open an issue on GitHub

---

**Ready to start?** Run: `./run_federated_learning.sh`

**Questions?** Check the detailed `README.md`

**Want to learn more?** Explore the Jupyter notebooks!
