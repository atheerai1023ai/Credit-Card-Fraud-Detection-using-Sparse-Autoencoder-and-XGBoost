<div align="center">

#  Predicting Thyroid Problems
### A Machine Learning Approach to Thyroid Disease Classification

*Turning routine lab panels into early, actionable clinical signal.*

[![Status](https://img.shields.io/badge/status-completed-brightgreen)]()
[![Task](https://img.shields.io/badge/task-multiclass%20classification-blue)]()
[![Best Model](https://img.shields.io/badge/best%20model-AdaBoost%20(weighted)-orange)]()
[![Accuracy](https://img.shields.io/badge/accuracy-98%25-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

</div>

---

##  Overview

Thyroid disorders are among the most common endocrine conditions, quietly affecting metabolism, energy regulation, and overall health — and they're easy to miss until they're serious. The clinical challenge is that the line between "normal" and "abnormal" thyroid function is subtle, and it depends on a wide combination of lab values, medical history, and demographics rather than any single marker.

This project builds a **multiclass classification model** that predicts whether a patient is **Hyperthyroid, Hypothyroid, or Negative (no thyroid condition)**, using a clinical dataset of lab results, medication history, and demographic features sourced from Kaggle.

The core question we're answering: *can a model catch the minority, harder-to-detect classes (especially Hypothyroid) reliably enough to be clinically useful — not just accurate on the easy majority class?*

---

##  Dataset

| | |
|---|---|
| **Source** | [Kaggle — Thyroid Disease Classification Dataset](https://www.kaggle.com/datasets/your-dataset-url) |
| **Size** | 9,172 rows × 31 columns |
| **Target classes** | `Negative` (majority), `Hypothyroid`, `Hyperthyroid` |
| **Duplicates** | None |
| **Missing data** | Present, concentrated in hormone measurements and `sex` |

### Feature groups

<details>
<summary><b> Full attribute list (click to expand)</b></summary>

**Demographics**
- `age` — patient age (int)
- `sex` — gender (string)

**Medical history** (all boolean unless noted)
- `on_thyroxine`, `query_on_thyroxine`, `on_antithyroid_meds`, `sick`, `pregnant`, `thyroid_surgery`, `I131_treatment`, `query_hypothyroid`, `query_hyperthyroid`, `lithium`, `goitre`, `tumor`, `psych`
- `hypopituitary` — float (dropped during preprocessing, see below)

**Laboratory test results** (each paired with a `*_measured` boolean flag)
- `TSH`, `T3`, `TT4`, `T4U`, `FTI`, `TBG`

**Other**
- `referral_source`, `target` (diagnosis), `patient_id`

</details>

A key structural insight surfaced during EDA: **missingness in lab values isn't random.** Whenever a `*_measured` flag is `'f'` (test not performed), the corresponding lab-value column is missing — the "missing" data is really "not tested," not noise.

---

##  Data Preprocessing & Feature Engineering

| Issue | Fix | Why |
|---|---|---|
| Missing lab values (`TSH`, `T3`, `TT4`, `T4U`, `FTI`, `TBG`) | Imputed with `0` | Assumed "not measured," consistent with the `*_measured` flag pattern |
| Missing `sex` values | Imputed with mode | Simple, low-risk for a near-balanced categorical field |
| `hypopituitary` column | **Dropped** (`data.drop('hypopituitary', axis=1, inplace=True)`) | Contained only a single unique value (`'f'`) — zero predictive signal |
| Unrealistic `age` values (e.g., 455, 65511, 65512, 65526) | Filtered: `data = data[data['age'] <= 100]` | Clear data-entry errors / outliers that would bias the model |
| Categorical variables | Label Encoding / One-Hot Encoding as appropriate | Standard ML-readiness prep |
| Target variable | Engineered into 3 classes: `Hyperthyroid`, `Hypothyroid`, `Negative` | Reframes the problem as clean multiclass classification |

**Sanity check that paid off:** no male patients were flagged `pregnant` — confirming the dataset respects basic biological constraints, which gave confidence in overall data integrity before modeling.

---

##  Exploratory Analysis — Key Findings

- **Gender skew**: Both hyperthyroidism and hypothyroidism occur noticeably more often in women than men.
- **Severe class imbalance**: The `Negative` class dominates the target distribution — this became the central challenge of the whole project.
- **Age**: Age distribution didn't show a strong discriminating pattern across thyroid conditions, though subtle trends may exist and warrant deeper study.
- **Self-report vs. diagnosis**: Comparing patient-reported thyroid conditions against actual diagnoses showed high agreement for `Negative` and `Hypothyroid` cases, but a meaningful share of self-reported conditions were actually diagnosed as `Negative` — a reminder that self-report is a weak proxy for ground truth.
- **Correlations**: A feature-correlation heatmap showed meaningful relationships among `TSH`, `TT4`, `T4U`, and `FTI` — consistent with their shared clinical role in thyroid function — and confirmed that most features carried real signal for the target.

---

##  Modeling Approach

Three algorithms were evaluated, all under an **80/20 stratified train/test split**:

| Model | Why it was chosen |
|---|---|
| **Decision Tree** | Interpretable, non-parametric baseline; handles mixed feature types natively |
| **Logistic Regression** | Simple linear baseline for probability-based classification |
| **AdaBoost** | Ensemble of weak learners (decision stumps), iteratively re-weights misclassified samples — a natural fit for an imbalanced target |

### Handling class imbalance — four AdaBoost variants tested

1. **Vanilla AdaBoost** — baseline ensemble, no imbalance correction
2. **Hyperparameter-tuned AdaBoost** — `GridSearchCV` over `n_estimators` and `learning_rate`
3. **AdaBoost + balanced class weights** — `class_weight='balanced'` on the base `DecisionTreeClassifier`
4. **AdaBoost + custom class weights** — hand-tuned per-class weights:
   - `Negative: 0.38` · `Hypothyroid: 4.26` · `Hyperthyroid: 8.17`

**Evaluation metric of choice: F1-score**, chosen deliberately over raw accuracy because (a) the classes are imbalanced, and (b) F1 balances precision and recall into one number that doesn't get fooled by a model that just predicts `Negative` every time.

---

##  Results

| Model | Accuracy | F1-Score | Notes |
|---|---|---|---|
| Naive Bayes | 87% | — | Struggled hardest with the imbalanced target; weakest on minority classes |
| Decision Tree | 98% | 92% | Strong across all three classes, especially `Hypothyroid` |
| AdaBoost (initial) | 98% | 91% | Strong overall, but weaker precision/recall on `Hypothyroid` before tuning |
| AdaBoost (`GridSearchCV`-tuned) | 98% | 91% | Tuning improved `Hypothyroid` precision & recall specifically |
| AdaBoost (balanced weights) | 97% | 90% | Recall on `Hypothyroid` improved, at the cost of `Negative`-class performance |
| **AdaBoost (custom weights)**  | **98%** | **91%** | **Best minority-class handling overall** — strongest `Hypothyroid` performance while keeping other classes strong |

###  Winning models
**AdaBoost (with customized class weights)** and **Decision Tree** emerged as the top performers, both landing ~98% accuracy — but AdaBoost's custom-weighted variant was the most *clinically* meaningful result, since it specifically improved detection of the harder-to-catch `Hypothyroid` class without sacrificing overall performance.

### Class-wise takeaways
- **Hyperthyroid**: Decision Tree and AdaBoost performed comparably, both clearly ahead of Naive Bayes.
- **Hypothyroid** (the hardest class): Customized-weight AdaBoost was the strongest — this is the headline result of the project.
- **Negative**: All models did well here, unsurprisingly, given it's the majority class.

---

##  Discussion

### Strengths
- Rigorous EDA surfaced real structural insights in the data (the `*_measured` ↔ missingness relationship, the pregnancy sanity check) before any modeling began — not just plots for their own sake.
- Missing values and data anomalies (the `age` outliers) were identified and handled with clear, justified reasoning rather than blanket imputation.
- Multiple models were evaluated with a consistent, imbalance-aware metric (F1-score), plus accuracy, precision, recall, Balanced Accuracy Rate (BAR), and Balanced Error Rate (BER) for a fuller picture.
- Hyperparameter tuning (`GridSearchCV`) was applied purposefully to address the specific weak point (minority-class detection) rather than generically maximizing accuracy.

### Limitations
- **Class imbalance** remains the dominant challenge — the `Negative` class heavily outnumbers `Hypothyroid`/`Hyperthyroid`, which risks biasing predictions toward the majority class even after weighting.
- **Limited feature engineering** — the project largely worked with existing features rather than engineering new derived ones; there's likely more signal to extract.
- **Generalizability is unconfirmed** — performance was evaluated on one dataset only; it hasn't been tested against an external population or dataset.
- **Small effective sample for minority classes** — with relatively few `Hypothyroid`/`Hyperthyroid` examples, reported metrics may be more volatile than they'd be at larger scale.

### Challenges → Solutions

| Challenge | Solution |
|---|---|
| Class imbalance | Class weighting (`balanced` and custom) in AdaBoost and Logistic Regression |
| Missing values | Zero-imputation for continuous hormone features, mode-imputation for `sex` |
| Data anomalies | Filtered biologically implausible `age` values (>100) |

---

##  Tech Stack

- **Language**: Python
- **Modeling**: `scikit-learn` (Decision Tree, Logistic Regression, AdaBoost, Naive Bayes, `GridSearchCV`)
- **Data handling**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`
- **Environment**: Jupyter Notebook

---

##  Repository Structure

```
.
├── prthyroid.ipynb     # Full pipeline: EDA → preprocessing → modeling → evaluation
├── .gitignore
└── README.md
```

---

##  Future Work

- Deeper feature engineering — derived/interaction features from correlated lab markers (`TSH`, `TT4`, `T4U`, `FTI`)
- Testing additional models (e.g., gradient boosting, random forest, XGBoost) under the same weighting strategy
- External validation on a second, independent thyroid dataset to test generalizability
- Collecting a larger sample of minority-class cases to stabilize minority-class metrics

---

##  References

1. **Kaggle** — [Thyroid Disease Classification Dataset](https://www.kaggle.com/datasets/your-dataset-url)
2. **CDC** — [Thyroid Disease](https://www.cdc.gov/thyroiddisease/index.html)
3. **American Thyroid Association** — [Understanding Thyroid Disease](https://www.thyroid.org/understanding-thyroid-disease/)

---



<div align="center">

*A small project with a real point: catching the class that's easy to miss matters more than the overall score.*

</div>
