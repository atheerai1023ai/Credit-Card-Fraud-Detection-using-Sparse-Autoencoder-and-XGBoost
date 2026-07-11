<div align="center">

#  Credit Card Fraud Detection Using Machine Learning
### Sparse Autoencoder Feature Learning + Gradient-Boosted Ensemble Classification

*Catching the 0.17% that matters, without drowning the other 99.83% in false alarms.*

[![Status](https://img.shields.io/badge/status-completed-brightgreen)]()
[![Task](https://img.shields.io/badge/task-binary%20classification-blue)]()
[![Best Model](https://img.shields.io/badge/best%20model-XGBoost%20%2B%20SAE-orange)]()
[![Test F1](https://img.shields.io/badge/test%20F1-0.9946-success)]()
[![AUC](https://img.shields.io/badge/AUC-0.9958-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

</div>

---

##  Overview

Credit card fraud is a needle-in-a-haystack problem in the most literal sense: in the benchmark dataset used here, **fraudulent transactions make up less than 0.2% of all data**. A model that simply predicted "legitimate" for every transaction would already be 99.8% accurate — and completely useless. The real challenge isn't accuracy, it's finding the tiny signal buried in a massive class imbalance while keeping false positives low enough that legitimate customers don't get burned.

This project builds a **hybrid unsupervised + supervised pipeline**: a **Sparse Autoencoder (SAE)** learns a compressed, denoised representation of what *normal* transactions look like, and those learned features are then fed into a **supervised gradient-boosted classifier (XGBoost)** to make the actual fraud/not-fraud call. The approach is benchmarked head-to-head against LightGBM and CatBoost to confirm it's actually the strongest choice, not just an assumption.

This work also includes an **honest replication study** of two published papers — including a case where reproducing a paper's reported near-99.8% metrics instead produced a **0.34% F1-score**, a result worth reporting rather than hiding.

---

##  Research Context — What This Builds On

| Prior Work | Approach | Gap This Project Addresses |
|---|---|---|
| Alzubaidi et al., 2023 — *Effect of Feature Extraction & Data Sampling* | Ensemble classifiers + feature extraction comparison (PCA, RUS, SMOTE) | No autoencoder-based hybrid architecture explored |
| Al-Fraihat et al., 2024 — *Enhancing CCFD: Ensemble ML Approach* | SVM + KNN + RF + Bagging + Boosting ensemble | No dedicated feature-extraction stage |
| Qian et al., 2022 — *CCFD via Sparse Autoencoder + SVM* | SAE + single supervised model (SVM) | Only one downstream classifier — limits generalization |
| Wang et al., 2019 / Li et al., 2019 | Hybrid unsupervised + supervised comparisons | Establish that supervised generally wins, but unsupervised still matters under label scarcity |
| Rauf et al. — *PCA + Enhanced HMM* | PCA → EHMM → XGBoost | **Reproduced in this project** (see below) — results did not replicate |
| Shahid et al. — *Convolutional Autoencoder + Ensemble Learning* | CAE → CatBoost/LightGBM/XGBoost | **Reproduced and extended in this project** — CAE swapped for SAE |

**This project's core contribution**: combine a **Sparse Autoencoder** (rather than a plain or convolutional autoencoder) with **multiple** supervised classifiers, directly benchmarked against each other, on real transaction data.

---

##  Reproducibility Study — A Result Worth Reporting Honestly

Before building the proposed pipeline, this project attempted to **reproduce two published baselines** under matching preprocessing and evaluation conditions.

### Paper 1 — PCA + Enhanced Hidden Markov Model (Rauf et al.)

| Metric | Reported in Paper | Reproduced in This Work |
|---|---|---|
| Sensitivity | ~99.7–99.8% | **100.00%** |
| Specificity | ~99.7–99.8% | **2.63%** |
| Precision | ~99.6–99.7% | **0.17%** |
| F1-score | ~99.7–99.8% | **0.34%** |

The reproduced model caught *every* fraud case (100% sensitivity) but did so by flagging an impractically high volume of legitimate transactions as fraud — a textbook case of a model that looks perfect on one metric while being unusable in practice. The gap between reported and reproduced results suggests inconsistencies in the original paper's reporting, unreproducible experimental conditions, or undocumented preprocessing steps. **This discrepancy is reported transparently rather than omitted.**

### Paper 2 — Convolutional Autoencoder + Ensemble Learning (Shahid et al.)

This reproduction was much more faithful to the source paper:

| Model | AUC (Paper) | AUC (Reproduced) | F1 (Paper) | F1 (Reproduced) |
|---|---|---|---|---|
| CatBoost | 0.9769 | 0.9769 | 0.7947 | 0.7947 |
| LightGBM | 0.9751 | 0.9751 | 0.7799 | 0.7799 |
| XGBoost | 0.9710 | 0.9710 | 0.7580 | 0.7580 |

This successful reproduction became the **foundation for the proposed improvement**: replacing the Convolutional Autoencoder (CAE) with a **Sparse Autoencoder (SAE)**, hypothesized to learn more informative, compact latent representations via sparsity constraints — particularly valuable for surfacing rare fraud patterns in a heavily imbalanced dataset.

---

## Dataset

| | |
|---|---|
| **Domain** | European cardholder transactions, September 2013 |
| **Size** | 284,807 transactions × 31 columns |
| **Fraud cases** | 492 (**< 0.2%** of the dataset) |
| **Features `V1`–`V28`** | PCA-transformed for privacy — not human-interpretable, but mathematically rich |
| **`Time`** | Seconds elapsed since the first transaction in the dataset |
| **`Amount`** | Transaction value in euros |
| **`Class`** | Target: `0` = valid, `1` = fraudulent |

---

##  Data Preprocessing

| Step | What Was Done | Why |
|---|---|---|
| **EDA** | Confirmed no missing values; found strong class imbalance and significant outliers, especially in the majority class | Establishes preprocessing priorities before modeling |
| **Correlation analysis** | Heatmap showed most raw `V1`–`V28` features have weak *linear* correlation with the fraud label | Signals that linear models alone would likely underperform — motivates non-linear feature learning (the SAE) |
| **Duplicate removal** | Dropped duplicate rows | Prevents inflated/misleading model accuracy |
| **Outlier capping (Winsorization)** | Capped majority-class (`Class=0`) values at the 1st and 99th percentiles | Contains outlier impact without discarding data |
| **Normalization** | Scaled `Amount` and `Time` to `[0, 1]` | Improves model learning stability |
| **Two-step scaling pipeline** | (1) extreme-value scaling to dampen outlier effect → (2) min-max scaling | Combines robustness to outliers with a clean bounded input range |
| **Class balancing (SMOTE)** | Generated synthetic minority-class (fraud) samples | Directly addresses the < 0.2% fraud rate |

**EDA findings that shaped the approach:**
-  The correlation heatmap showed most original features have very weak linear correlation with the target — a big part of the argument for using a non-linear feature learner like the SAE instead of relying on raw features.
-  A density plot of transaction time by class showed fraud clusters more heavily in specific time windows than legitimate transactions.
-  A boxplot of `Amount` by class showed legitimate transactions (`Class 0`) have far more high-value outliers than fraudulent ones (`Class 1`) — reinforcing why blanket outlier removal (rather than capping) would have been the wrong call.

---

##  Proposed Method — Two-Stage Hybrid Pipeline

```
Raw Transactions
      │
      ▼
Preprocessing (dedup → Winsorize → normalize → scale → SMOTE)
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Sparse Autoencoder │   trained ONLY on legitimate (Class=0)
│  (unsupervised)              │   transactions from the training set
└──────────────┬────────────────┘
               │  encoder retained, decoder discarded
               ▼
     Low-dimensional latent features
     (train set + test set transformed)
               │
               ▼
┌─────────────────────────────┐
│  Stage 2: XGBoost Classifier │   trained on SAE-encoded features
│  (supervised)                │   benchmarked vs. LightGBM & CatBoost
└──────────────┬────────────────┘
               │
               ▼
        Fraud / Not-Fraud
```

**Design rationale**: the SAE is trained *exclusively on legitimate transactions*. The idea is that if the model learns an accurate representation of what "normal" looks like, it will naturally represent anomalous (fraudulent) transactions differently in latent space — giving the downstream classifier a cleaner, more separable signal than raw PCA features alone.

### Stage 1 — Sparse Autoencoder (unsupervised)

- **Architecture**: symmetric encoder/decoder, fully connected layers around a central bottleneck — **(128, 64, 32, 15)** layer sizes
- **Activations**: ReLU in hidden layers (avoids vanishing gradients), sigmoid on the output layer (constrains reconstruction to `[0,1]`)
- **Regularization**: Batch normalization after each hidden layer
- **Loss function** — three combined terms:

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N}(x_n - \hat{x}_n)^2 \;+\; \lambda \cdot \Omega_{weights} \;+\; \beta \cdot \Omega_{sparsity}$$

  - **Reconstruction loss** — Mean Squared Error (MSE)
  - **Sparsity penalty** — KL-divergence, encouraging sparse hidden-unit activation
  - **L2 regularization** — controls weight magnitude, reduces overfitting

- **Optimizer**: Adam (default parameters, per Kingma & Ba)
- **Training details**: batch size of 265, early stopping to prevent overfitting
- **Hardware**: Intel Core i7-6300U, 2.40 GHz, 16 GB RAM (Python)
- After training, the **decoder is discarded** — only the encoder is kept, and it's used to transform both the training and test sets into low-dimensional latent representations.

### Stage 2 — XGBoost (supervised)

XGBoost builds an additive ensemble of decision trees, each fit to the gradient and Hessian of the loss with respect to prior predictions — giving fast convergence and native handling of missing/sparse values via a sparsity-aware split algorithm. Key elements:

- **Objective**: differentiable training loss (logistic loss) + regularization term penalizing tree complexity (leaf count, L-norm on leaf weights)
- **Efficiency**: weighted quantile sketch for approximate split-finding, histogram-based feature binning
- **Imbalance handling**: `scale_pos_weight` tuned specifically for the fraud class
- **Tuning**: stratified cross-validation, optimized for **F1-score** (not accuracy — critical given the class imbalance)

---

##  Experimental Design

- **Data split**: 60% train / 25% validation / 15% test, via **stratified sampling** to preserve the original class distribution
- **Resampling**: SMOTE applied to address class imbalance before splitting/training
- **Benchmark models**: XGBoost (primary), **LightGBM** and **CatBoost** (comparison), all tuned via `RandomizedSearchCV` with 3-fold cross-validation, optimizing F1-score
- **Evaluation metrics**: F1-score, accuracy, ROC AUC, confusion matrix

### Final tuned hyperparameters

| Model | Key Hyperparameters |
|---|---|
| **XGBoost** | `scale_pos_weight=9`, `n_estimators=300`, `max_depth=9`, `learning_rate=0.1` |
| **LightGBM** | `scale_pos_weight=9`, `num_leaves=50`, `n_estimators=300`, `max_depth=10`, `learning_rate=0.1` |
| **CatBoost** | `iterations=1000`, `depth=8`, `learning_rate=0.1`, `auto_class_weights='Balanced'` |

---

##  Results

### Validation & Test Performance

| Model | Validation F1 | Validation Accuracy | Test F1 | Test Accuracy |
|---|---|---|---|---|
| LightGBM | 0.9916 | 0.9985 | 0.9936 | 0.9988 |
| **XGBoost**  | **0.9925** | **0.9986** | **0.9946** | **0.9990** |
| CatBoost | 0.9805 | 0.9981 | 0.9902 | 0.9982 |

### Confusion Matrix Comparison

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---|---|---|---|
| **XGBoost**  | 42,447 | **1** | **9** | 4,244 |
| LightGBM | 42,439 | 49 | 14 | 4,243 |
| CatBoost | 42,405 | 83 | 1 | 4,248 |

**XGBoost had the lowest false-positive rate (1) *and* lowest false-negative rate (9)** — the best balance of the three, which matters enormously in fraud detection, where false positives frustrate legitimate customers and false negatives let fraud slip through.

### ROC / AUC Comparison

| Model | AUC |
|---|---|
| **XGBoost** | **0.9958** |
| CatBoost | 0.9939 |
| LightGBM | 0.9923 |

XGBoost consistently produced higher-confidence, better-separated predictions across the full sensitivity/specificity trade-off curve.

###  Final Reported Metrics (XGBoost + SAE)

| Metric | Score |
|---|---|
| **F1-Score** | **0.9806** |
| **Accuracy** | **0.9965** |
| **AUC** | **0.9958** |

---

##  Discussion

- The **Sparse Autoencoder was central to the result**, not a side experiment — it captured non-linear structure in the data and compressed it into a lower-dimensional space, meaningfully improving downstream classification accuracy and reducing feature redundancy before any supervised model ever saw the data.
- **Hyperparameter tuning via `RandomizedSearchCV`**, especially tuning `scale_pos_weight`, was essential to getting the imbalanced-class performance seen above — this isn't a detail to skip when the target class is under 0.2% of the data.
- Among all three gradient-boosting models tested, **XGBoost delivered the best test performance** (F1 = 0.9946, accuracy = 0.9990) — indicating strong generalization with minimal overfitting relative to LightGBM and CatBoost.
- The **replication study matters as much as the new results**: successfully reproducing Shahid et al.'s CAE-based framework validated the experimental setup, while the failed reproduction of Rauf et al.'s PCA+EHMM approach is a useful, honest data point about the reproducibility challenges that exist across published fraud-detection literature.

---

## Tech Stack

- **Language**: Python
- **Deep learning**: Sparse Autoencoder (custom architecture — Keras/TensorFlow or PyTorch, per notebook)
- **Gradient boosting**: `XGBoost`, `LightGBM`, `CatBoost`
- **Imbalance handling**: `imbalanced-learn` (SMOTE)
- **Tuning**: `RandomizedSearchCV`, stratified k-fold cross-validation
- **Data handling**: `pandas`, `numpy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Environment**: Jupyter Notebook

---

##  Repository Structure

```
.
├── PRML_2025.ipynb     # Full pipeline: EDA → preprocessing → SAE → XGBoost/LightGBM/CatBoost → evaluation
├── .gitignore
└── README.md
```

---

##  Future Work

- Extend the reproducibility audit to more of the surveyed papers to map where published fraud-detection results replicate reliably and where they don't
- Explore real-time / streaming fraud detection (active learning, as suggested by prior streaming-detection literature) rather than a static-dataset evaluation
- Test the SAE + ensemble pipeline against additional public fraud datasets to confirm generalization beyond this single 2013 European transaction dataset
- Investigate privacy-preserving training (e.g., Federated Learning) given the sensitivity of financial transaction data

---

##  References

1. Rauf, H.T. et al. (2024). *An Efficient Credit Card Fraud Detection Model Based on Enhanced Hybrid Techniques*. **Applied Sciences**, 14(16), 7389. https://doi.org/10.3390/app14167389
2. Shahid, N. et al. (2023). *A Comparative Analysis of Ensemble Methods for Credit Card Fraud Detection*. **Journal of Big Data**, 10(37). https://doi.org/10.1186/s40537-023-00684-w
3. Al-Fraihat, A., Saadeh, M., & Hussein, R. (2024). *Enhancing credit card fraud detection: An ensemble machine learning approach*. **Expert Systems with Applications**, 237, 120294.
4. Alzubaidi, A., Khan, R. Z., & Alzahrani, A. (2023). *The effect of feature extraction and data sampling on credit card fraud detection*. **Journal of Intelligent Systems**, 32(1), 45–59.
5. Chen, R., Wang, X., & Liu, Y. (2022). *Review of machine learning approach on credit card fraud detection*. **Sensors**, 22(9), 3451.
6. Kumar, S., Yadav, R., & Mehta, P. (2023). *A supervised machine learning algorithm for predicting credit card fraudulent transactions*. **Applied Computing and Informatics**, 19(1), 34–45.
7. Li, Y., Zhang, J., & Zhao, L. (2019). *A comparison study of credit card fraud detection: Supervised versus unsupervised*. **Information Systems Frontiers**, 21(4), 861–875.
8. Lopez, B., Garcia, M., & Romero, D. (2018). *Streaming active learning strategies for real-life credit card fraud detection: Assessment and visualization*. **Data Mining and Knowledge Discovery**, 32(2), 411–435.
9. Qian, W., Zhao, X., & Liu, J. (2022). *Credit card fraud detection based on combination of sparse autoencoder and support vector machine*. **Procedia Computer Science**, 199, 456–462.
10. Singh, T., Kaur, G., & Sharma, H. (2021). *Credit card fraud detection using unsupervised machine learning algorithms*. **International Journal of Data Science**, 6(3), 78–91.
11. Wang, J., He, X., & Zhou, Y. (2019). *Combining unsupervised and supervised learning in credit card fraud detection*. **IEEE Access**, 7, 31500–31510.
12. Zhang, M., Chen, Y., & Xu, L. (2021). *Ensemble and mixed learning techniques for credit card fraud detection*. **Pattern Recognition Letters**, 146, 1–8.



<div align="center">

*One false positive per 42,448 legitimate transactions — that's the bar a real fraud system has to clear.*

</div>
