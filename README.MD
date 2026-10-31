# Bank Customer Churn Prediction

A Machine Learning classification project that predicts whether a bank customer is likely to leave the bank using historical customer data.

The project explores data preprocessing, class imbalance handling, model training, hyperparameter tuning, evaluation metrics, and business insights to understand how machine learning can support customer retention strategies.

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Model Evaluation](#model-evaluation)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Feature Importance](#feature-importance)
- [Business Recommendations](#business-recommendations)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

## 📖 Overview

Customer churn is a major challenge for banks because losing customers can affect revenue and long-term customer relationships.

This project uses supervised machine learning to identify customers who may leave the bank.

Two classification approaches were evaluated:

1. Logistic Regression with SMOTE
2. Decision Tree with SMOTE

After comparing their performance, hyperparameter tuning was performed on the Decision Tree using a validation set.

The final Decision Tree achieved an ROC-AUC score of **0.8346** on the test set.

## 🎯 Problem Statement

Banks have historical information about their customers, but identifying which customers are likely to leave can be challenging.

The goal of this project is to develop a binary classification model that predicts customer churn based on customer attributes.

The target variable is `Exited`:

- `0` — Customer did not leave the bank.
- `1` — Customer left the bank.

The predictions could help banks identify potentially high-risk customers and develop targeted retention strategies.

## 🎯 Objectives

- Explore and understand customer data.
- Perform data cleaning and preprocessing.
- Handle class imbalance using SMOTE.
- Build a Logistic Regression baseline model.
- Train and evaluate a Decision Tree classifier.
- Compare models using multiple evaluation metrics.
- Tune Decision Tree hyperparameters.
- Analyze feature importance.
- Interpret model performance from a business perspective.

## 📊 Dataset

**Dataset:** `Churn_Modelling.csv`

The dataset contains 10,000 customer records and 14 original columns.

| Feature | Description |
|---|---|
| RowNumber | Row identifier |
| CustomerId | Customer identifier |
| Surname | Customer surname |
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Customer's estimated salary |
| Exited | Target variable indicating churn |

The columns `RowNumber`, `CustomerId`, and `Surname` were excluded from the model inputs.

**Note:** The dataset file is not included in this repository by default. Add it locally if you have permission to redistribute it.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

## 🔄 Project Workflow

The project follows this workflow:

1. Load and inspect the dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Separate features and target variable.
4. Split data into training and testing sets.
5. Preprocess numerical and categorical features.
6. Train Logistic Regression with SMOTE.
7. Train Decision Tree with SMOTE.
8. Compare model performance.
9. Tune Decision Tree hyperparameters using validation data.
10. Retrain the selected model on the complete training data.
11. Evaluate the final model on the test set.
12. Analyze feature importance and business implications.

## 🧹 Data Preprocessing

### 1. Feature and target separation

The target variable is `Exited`.

The following columns were removed from the input features:

- `RowNumber`
- `CustomerId`
- `Surname`

These columns were not used as predictive inputs in this project.

### 2. Train-test split

The dataset was split into:

- 80% training data
- 20% testing data

Stratified splitting was used to preserve the approximate target-class distribution.

A separate validation set was created from the training data for hyperparameter selection.

### 3. Numerical feature preprocessing

Numerical features were processed using:

- **Median Imputation:** Replaces missing numerical values with the median.
- **StandardScaler:** Standardizes numerical features.

### 4. Categorical feature preprocessing

Categorical features were processed using:

- **Most-Frequent Imputation:** Replaces missing values with the most frequent category.
- **One-Hot Encoding:** Converts categorical variables into numerical features.

A `ColumnTransformer` and Scikit-learn pipeline were used to organize preprocessing and model training.

### 5. Handling class imbalance with SMOTE

The target classes are imbalanced, with fewer customers in the churn class.

SMOTE (Synthetic Minority Over-sampling Technique) was used to generate synthetic minority-class examples during training.

SMOTE was included inside the training pipeline to avoid applying oversampling to the held-out test set.

## 🤖 Models Implemented

### 1. Logistic Regression + SMOTE

Logistic Regression was selected as the baseline model because it is a commonly used approach for binary classification.

The model was trained with:

- Numerical and categorical preprocessing
- SMOTE for class imbalance handling
- Logistic Regression as the classifier

### 2. Decision Tree + SMOTE

A Decision Tree classifier was trained to learn feature-based decision rules.

The initial model used a maximum depth of 5.

Its performance was compared against the Logistic Regression baseline.

### 3. Hyperparameter Tuning

The Decision Tree was tuned using a validation set.

The following parameter values were evaluated:

| Hyperparameter | Values |
|---|---|
| `max_depth` | 5, 6, 7, 9 |
| `min_samples_split` | 2, 10, 20 |
| `min_samples_leaf` | 1, 5, 10 |

A total of 36 parameter combinations were evaluated.

The selected configuration was:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

The candidate was selected based on validation performance, then retrained using the complete training data before final test evaluation.

## 📈 Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

### Model Comparison

| Metric | Logistic Regression + SMOTE | Decision Tree + SMOTE |
|---|---:|---:|
| Accuracy | 0.7065 | 0.7765 |
| Precision | 0.3774 | 0.4695 |
| Recall | 0.6806 | 0.7568 |
| F1 Score | 0.4855 | 0.5795 |
| ROC-AUC | 0.7501 | 0.8303 |

The initial Decision Tree outperformed Logistic Regression on all five reported metrics.

### Final Tuned Decision Tree Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7775 |
| Precision | 0.4712 |
| Recall | 0.7641 |
| F1 Score | 0.5829 |
| ROC-AUC | 0.8346 |

### Metric Interpretation

- **Accuracy — 77.75%:** The proportion of test predictions classified correctly.
- **Precision — 47.12%:** Among customers predicted to churn, the proportion who actually churned.
- **Recall — 76.41%:** The proportion of actual churners correctly identified.
- **F1 Score — 0.5829:** The harmonic mean of precision and recall.
- **ROC-AUC — 0.8346:** Measures how well the model ranks churners above non-churners across different classification thresholds.

The model identifies a substantial proportion of actual churners, although its precision indicates that many predicted churners do not actually churn.

## 🔍 Confusion Matrix Analysis

The final model produced the following confusion matrix on the test set:

```text
[[1244, 349],
 [  96, 311]]
```

| Actual / Predicted | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 1244 | 349 |
| Actual Churn | 96 | 311 |

### Interpretation

- **True Negatives (1244):** Non-churning customers correctly identified.
- **False Positives (349):** Non-churning customers incorrectly predicted to churn.
- **False Negatives (96):** Actual churners missed by the model.
- **True Positives (311):** Actual churners correctly identified.

Out of 407 actual churners in the test set, the model correctly identified 311.

From a business perspective, false negatives represent customers who may leave without being flagged, while false positives may lead to unnecessary retention offers.

## 🌳 Feature Importance

Feature importance was extracted from the trained Decision Tree using its `feature_importances_` attribute.

The transformed feature names were obtained from the fitted preprocessor, allowing importance scores to be associated with the corresponding numerical or encoded categorical features.

A sorted table and bar chart were used to examine the top 10 features.

**Important:** Feature importance reflects how features contribute to impurity reduction within the trained tree. It does not establish causation or prove that a feature directly causes customer churn.

The exact ranking of important features depends on the trained model and dataset.

## 💼 Business Recommendations

Based on the model's predictions, a bank could explore the following strategies:

1. **Identify potentially high-risk customers:** Use predicted churn probabilities to prioritize customers for further review.
2. **Develop targeted retention campaigns:** Test appropriate offers or customer-support interventions for selected high-risk groups.
3. **Monitor false positives:** Estimate the cost of contacting customers who would not have churned.
4. **Reduce missed churners:** Evaluate classification thresholds using validation data when missing a potential churner is particularly costly.
5. **Measure campaign effectiveness:** Determine whether targeted interventions actually improve customer retention.
6. **Monitor model performance:** Evaluate predictions on newer data to identify changes in customer behaviour or data distributions.

These recommendations are potential applications, not proven business outcomes. Their effectiveness should be validated through real-world experiments.

## ⚠️ Limitations

- The model was developed using a historical dataset, so its performance may differ on newer customer data.
- Precision is relatively low at 47.12%, meaning many predicted churners are not actual churners.
- Feature importance does not establish causal relationships.
- The model's predictions do not guarantee that a retention offer will prevent churn.
- The selected classification threshold may not be optimal for every business objective.
- The project does not yet demonstrate the financial impact of a retention campaign.

## 🚀 Future Improvements

- Compare Decision Tree with Random Forest and Gradient Boosting models.
- Evaluate probability calibration and tune classification thresholds using validation data.
- Use cross-validation for more robust model selection.
- Investigate model interpretability using SHAP or permutation importance.
- Perform cost-sensitive evaluation using estimated retention costs and customer value.
- Test the model on more recent customer data.
- Build an interactive prediction interface using Streamlit or Flask.
- Deploy the model through an API and monitor performance over time.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd bank-customer-churn-prediction
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter
```

### 4. Add the dataset

Place `Churn_Modelling.csv` in the location expected by your notebook.

### 5. Run the notebook

```bash
jupyter notebook
```

Open your project notebook and execute its cells in order.

**Note:** The commands above assume the notebook contains the code used in this project. Update the dataset path and notebook filename to match your repository.

## 📁 Project Structure

An example structure for organizing the project:

```text
bank-customer-churn-prediction/
│
├── data/
│   └── Churn_Modelling.csv
│
├── notebooks/
│   └── bank_customer_churn_prediction.ipynb
│
├── images/
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

Adjust the structure to match your actual repository. Do not commit private customer data or files you are not authorized to redistribute.

## 👨‍💻 Author

**Arish Ahmad**

Machine Learning project focused on customer churn prediction, classification evaluation, and business-oriented model interpretation.

---

⭐ If you find this project useful, consider starring the repository.
