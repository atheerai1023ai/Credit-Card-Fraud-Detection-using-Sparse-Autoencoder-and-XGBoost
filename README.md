# Credit-Card-Fraud-Detection-using-Sparse-Autoencoder-and-XGBoost
A machine learning framework for credit card fraud detection using Sparse Autoencoder-based feature extraction and XGBoost classification. The project addresses the challenge of highly imbalanced financial transaction data through preprocessing, feature learning, and model optimization.
# Credit Card Fraud Detection using Sparse Autoencoder and XGBoost

## Overview

Credit card fraud detection is one of the most challenging problems in machine learning due to the extreme imbalance between legitimate and fraudulent transactions. Detecting fraud accurately while minimizing false alarms is essential for modern financial systems.

This project presents a hybrid machine learning framework that combines unsupervised feature learning using a Sparse Autoencoder (SAE) with supervised classification models. The extracted features are evaluated using XGBoost, LightGBM, and CatBoost to identify the most effective approach for fraud detection.

The implementation includes data preprocessing, feature engineering, model training, hyperparameter optimization, and comprehensive evaluation on a real-world credit card transaction dataset.

---

## Problem Statement

Fraudulent transactions account for less than 0.2% of all transactions in the dataset, making this a highly imbalanced classification problem. Traditional machine learning models often struggle to detect these rare events without generating excessive false positives.

This project addresses these challenges through:

- Data preprocessing and normalization
- Handling class imbalance using SMOTE
- Sparse Autoencoder (SAE) feature extraction
- Gradient boosting classifiers
- Hyperparameter optimization
- Performance comparison across multiple models

---

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Duplicate removal
- Outlier handling using Winsorization
- Feature scaling and normalization
- SMOTE data balancing
- Sparse Autoencoder feature extraction
- XGBoost classifier
- LightGBM classifier
- CatBoost classifier
- Model evaluation and comparison

---

## Dataset

The project uses the publicly available **Credit Card Fraud Detection** dataset containing **284,807 real-world transactions**, of which only **492 are fraudulent**. This dataset is widely used as a benchmark for fraud detection research because of its severe class imbalance. :contentReference[oaicite:2]{index=2}

---

## Technologies

- Python
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- LightGBM
- CatBoost
- Pandas
- NumPy
- Matplotlib

---

## Results

After feature extraction and model optimization, the models were evaluated using Accuracy, F1-score, ROC-AUC, and confusion matrices.

Among the evaluated models, **XGBoost achieved the best overall performance**, obtaining:

- Accuracy: **99.65%**
- ROC-AUC: **0.9958**
- F1-score: **0.9806**

These results demonstrate the effectiveness of combining Sparse Autoencoder feature learning with gradient boosting for highly imbalanced fraud detection problems. :contentReference[oaicite:3]{index=3}

---

## Repository Contents

- Machine learning implementation
- Data preprocessing pipeline
- Feature extraction model
- Model training scripts
- Evaluation notebooks
- Performance visualization

---

## Future Improvements

- Real-time fraud detection
- Transformer-based anomaly detection
- Explainable AI (XAI)
- Online learning for streaming transactions
- Deployment as a REST API

---

## License

This project is licensed under the MIT License.
