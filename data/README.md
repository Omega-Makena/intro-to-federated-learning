# Credit Card Fraud Detection Dataset

This directory is for storing the credit card fraud detection dataset.

## Dataset Information

### Option 1: Using Real Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:
- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **File**: `creditcard.csv`
- **Size**: ~150 MB
- **Samples**: 284,807 transactions
- **Features**: 30 (Time, V1-V28, Amount)
- **Target**: Class (0 = legitimate, 1 = fraud)

**To use the real dataset:**
1. Download `creditcard.csv` from Kaggle
2. Place it in this directory
3. Update the code to use: `filepath="data/creditcard.csv"`

### Option 2: Using Synthetic Data (Default)

For demonstration purposes, the code automatically generates synthetic data that mimics the structure of credit card transactions. This is useful when:
- You don't have access to the real dataset
- You want to quickly test the federated learning implementation
- You're learning about the concepts

The synthetic data generation is already implemented in `src/data_loader.py`.

## Dataset Structure

The dataset should have the following structure:

```
Time,V1,V2,V3,...,V28,Amount,Class
0,1.2,0.5,-1.3,...,0.8,149.62,0
406,2.1,-0.3,1.5,...,-0.4,2.69,0
...
```

**Columns:**
- `Time`: Seconds elapsed between this transaction and first transaction
- `V1-V28`: Principal components obtained with PCA (anonymized)
- `Amount`: Transaction amount
- `Class`: 0 (legitimate) or 1 (fraud)

## Data Privacy in Federated Learning

In our federated learning implementation:
- ✅ Each client keeps its local data in this directory
- ✅ Data **never** leaves the client's device
- ✅ Only model updates (weights/gradients) are sent to the server
- ✅ The server never sees the raw transaction data

This approach maintains privacy while enabling collaborative model training!

## Data Split

The data is split in two ways:

### 1. Train/Test Split
- Training: 80% of data
- Testing: 20% of data
- Stratified split to maintain class balance

### 2. Federated Split (among clients)
- **IID (Independent and Identically Distributed)**:
  - Data is randomly distributed among clients
  - Each client has similar data distribution
  
- **Non-IID**:
  - Data is distributed non-uniformly
  - Each client may have different class distributions
  - More realistic for real-world federated learning

## Class Imbalance

The fraud detection dataset is highly imbalanced:
- Legitimate transactions: ~99.8%
- Fraudulent transactions: ~0.2%

**Handling strategies:**
1. Weighted loss function (gives more weight to fraud class)
2. Weighted random sampling (balanced batches)
3. Evaluation with precision, recall, F1-score (not just accuracy)

## Usage in Code

```python
from data_loader import (
    load_and_preprocess_data,  # For real dataset
    create_synthetic_data,      # For synthetic dataset
    split_data_federated
)

# Option 1: Load real dataset
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
    filepath="data/creditcard.csv"
)

# Option 2: Create synthetic dataset
X, y = create_synthetic_data(n_samples=10000, fraud_ratio=0.002)

# Split among clients
client_data = split_data_federated(X_train, y_train, num_clients=3, iid=True)
```

## Notes

- The `.gitignore` file is configured to exclude `*.csv` files from version control
- This prevents accidentally committing large datasets to the repository
- Each user should download their own copy of the dataset
