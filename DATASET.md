# Dataset Information

## What Dataset Does This Project Use?

**Answer: By default, this project uses SYNTHETIC credit card transaction data.**

### Why Synthetic Data?

The synthetic data approach allows you to:
-  **Run immediately** - No dataset download required
-  **Learn quickly** - Get started in minutes
-  **Experiment freely** - No storage or bandwidth concerns
-  **Understand concepts** - Focus on federated learning, not data preparation

### Synthetic Data Characteristics

The automatically generated synthetic data has:
- **10,000 transactions** (configurable in code)
- **30 features**: Time, V1-V28 (PCA components), Amount
- **~0.2% fraud rate** (20 fraudulent transactions per 10,000)
- **Same structure** as the real Credit Card Fraud Detection dataset
- **Class imbalance** reflecting real-world fraud detection challenges

### Where Is the Synthetic Data Generated?

The synthetic data is generated at runtime in:
- **File**: `src/data_loader.py`
- **Function**: `create_synthetic_data()`
- **Usage**: Automatically called when you run `src/client.py` or `src/server.py`

```python
# From src/data_loader.py
def create_synthetic_data(n_samples=10000, n_features=30, fraud_ratio=0.002, random_state=42):
    """
    Create synthetic credit card transaction data for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate (default: 10,000)
        n_features (int): Number of features (default: 30)
        fraud_ratio (float): Ratio of fraudulent transactions (default: 0.002 = 0.2%)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X, y) features and labels
    """
    # Generates realistic synthetic data mimicking credit card transactions
```

## Can I Use a Real Dataset?

**Yes!** You can optionally use the real Credit Card Fraud Detection dataset from Kaggle.

### Real Dataset: Credit Card Fraud Detection (Kaggle)

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Characteristics**:
- **284,807 transactions** from European cardholders (September 2013)
- **492 frauds** (0.172% of all transactions)
- **30 features**:
  - `Time`: Seconds elapsed between each transaction and first transaction
  - `V1-V28`: Principal components obtained with PCA (anonymized for privacy)
  - `Amount`: Transaction amount
  - `Class`: 0 = legitimate, 1 = fraud
- **Size**: ~150 MB CSV file
- **Published by**: Worldline and Université Libre de Bruxelles (ULB)

### How to Use the Real Dataset

1. **Download the dataset**:
   - Go to [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`

2. **Place in data directory**:
   ```bash
   mv creditcard.csv /path/to/intro-to-federated-learning/data/
   ```

3. **Modify the code**:
   
   In `src/client.py`, line ~254, change:
   ```python
   # FROM (synthetic data):
   X, y = create_synthetic_data(n_samples=10000, fraud_ratio=0.002)
   
   # TO (real dataset):
   from data_loader import load_and_preprocess_data
   X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
       filepath="data/creditcard.csv"
   )
   # Then skip the train_test_split that follows
   ```

4. **Run normally**:
   ```bash
   python src/server.py
   python src/client.py --client-id 0
   ```

## Comparison: Synthetic vs Real Dataset

| Feature | Synthetic Data | Real Kaggle Dataset |
|---------|---------------|---------------------|
| **Setup** | None required | Download + configure |
| **Size** | Memory only | 150 MB file |
| **Transactions** | 10,000 (default) | 284,807 |
| **Fraud Rate** | ~0.2% | 0.172% |
| **Features** | 30 | 30 |
| **Structure** | Same as real | Real data |
| **Privacy** | Completely synthetic | Anonymized real data |
| **Use Case** | Learning & testing | Research & production |
| **Speed** | Instant | Requires download |

## Which Should You Use?

### Use Synthetic Data (Default) When:
-  Learning federated learning concepts
-  Testing the implementation quickly
-  Developing new features
-  Running demos or tutorials
-  You don't have access to Kaggle
-  You want zero setup time

### Use Real Dataset When:
-  Publishing research results
-  Comparing with other papers
-  Benchmarking performance
-  Validating on real-world data patterns
-  Production deployment preparation

## Data Privacy in Federated Learning

Regardless of which dataset you use (synthetic or real):

 **Data stays local** - Each client keeps its own data
 **Privacy preserved** - Only model updates are shared
 **Server doesn't see data** - Raw transactions never leave clients
 **Compliant** - Meets privacy regulations (GDPR, etc.)

This is the core benefit of federated learning!

## References

### Synthetic Data
- **Implementation**: `src/data_loader.py` - `create_synthetic_data()`
- **Usage**: Automatically called by `src/client.py`

### Real Dataset
- **Kaggle**: https://www.kaggle.com/mlg-ulb/creditcardfraud
- **Paper**: [Calibrating Probability with Undersampling for Unbalanced Classification](https://doi.org/10.1109/CIDM.2013.6597217)
- **Authors**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi

## Questions?

- **"Do I need to download anything?"** - No! Synthetic data works out of the box.
- **"Is the synthetic data realistic?"** - Yes, it mimics the structure and characteristics of real fraud detection data.
- **"Can I change the fraud rate?"** - Yes, edit the `fraud_ratio` parameter in `create_synthetic_data()`.
- **"How do I switch to the real dataset?"** - See instructions above or check `data/README.md`.

## Summary

 **Quick Answer**: This project uses **synthetic credit card transaction data by default**. You can optionally use the real Kaggle dataset if you want to work with actual fraud detection data.

 **For More Details**: See [`data/README.md`](data/README.md)
