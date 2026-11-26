# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The objective is to identify customers who are likely to discontinue their services and understand the factors associated with customer churn.

## Project Overview

Customer churn is an important challenge for subscription-based businesses. Identifying customers who are likely to leave allows companies to develop targeted retention strategies and improve customer retention.

This project explores customer data, preprocesses numerical and categorical features, trains multiple classification algorithms, and evaluates their performance to identify the most suitable model for churn prediction.

## Tech Stack

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** XGBoost
- **Deep Learning:** TensorFlow / Keras

## Dataset

**IBM Telco Customer Churn Dataset**

- Initial dataset size: 7,043 customer records
- Features: 20 input columns, including customer demographics, account information, subscribed services, and billing details.
- Target variable: `Churn`

Target encoding:

- `0` – Customer did not churn
- `1` – Customer churned

The dataset contained approximately 73.5% non-churn customers and 26.5% churn customers.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand customer characteristics and identify patterns associated with churn.

Key observations:
- Month-to-month contract customers showed higher churn.
- Customers with shorter tenure were more likely to churn.
- Customers who churned generally had higher monthly charges.

### 2. Data Cleaning and Preprocessing

- Converted `TotalCharges` into numerical format.
- Handled missing values by removing 11 records with invalid `TotalCharges`.
- Removed the `customerID` column.
- Encoded the target variable.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.

A `ColumnTransformer` was used to combine preprocessing steps.

### 3. Train-Validation-Test Split

The dataset was divided using stratified sampling:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Preprocessing was fitted only on the training data to prevent data leakage.

### 4. Model Development

Four classification algorithms were implemented and compared:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Neural Network using TensorFlow/Keras

The models were trained on the training dataset and compared using the same validation dataset.

### 5. Model Evaluation

The following evaluation metrics were used:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churned.
- **Recall:** Proportion of actual churners correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability of the model to distinguish between churners and non-churners.
- **Average Precision:** Summary measure of the Precision-Recall curve.

## Model Comparison

Validation performance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.37% | 69.39% | 60.50% | 64.64% | 85.32% |
| Random Forest | 78.67% | 63.59% | 46.62% | 53.80% | 81.99% |
| XGBoost | 81.23% | 68.95% | 53.74% | 60.40% | 84.73% |
| Neural Network | 80.85% | 68.90% | 51.25% | 58.78% | 85.16% |

Logistic Regression achieved the best overall validation performance and was selected for final evaluation.

## Final Model Performance

The selected Logistic Regression model was evaluated on the unseen test dataset.

| Metric | Test Score |
|---|---:|
| Accuracy | 78.86% |
| Precision | 61.92% |
| Recall | 52.86% |
| F1-Score | 57.03% |
| ROC-AUC | 82.21% |
| Average Precision | 58.00% |

### Confusion Matrix

```text
[[684  91]
 [132 148]]
```

The model correctly identified 148 customers who churned, while missing 132 actual churners.

### Evaluation Visualizations

The project includes:

- Model performance comparison chart
- Confusion matrix
- ROC curve
- Precision-Recall curve

These visualizations provide insights into model performance and the trade-off between precision and recall.

## Key Findings

- Logistic Regression outperformed the other three models on the validation set across accuracy, precision, recall, and F1-score.
- XGBoost delivered competitive performance.
- The Neural Network achieved an ROC-AUC close to Logistic Regression but had lower recall.
- More complex models did not necessarily produce better results.
- The final model demonstrated useful predictive ability, but there is room for improvement in identifying customers at risk of churn.

## Future Improvements

Potential areas for further improvement include:

- Classification threshold optimization.
- Handling class imbalance.
- Hyperparameter tuning.
- Feature engineering.
- Model interpretability and feature importance analysis.
- Deployment as a customer churn prediction application.

## Conclusion

This project provided practical experience in developing an end-to-end machine learning pipeline, from exploratory data analysis and preprocessing to model training, comparison, and evaluation.

An important finding was that Logistic Regression outperformed more complex algorithms on this dataset, demonstrating that model selection should be guided by evaluation results rather than algorithmic complexity.

The project also highlighted the importance of selecting appropriate evaluation metrics, particularly precision and recall, when working with imbalanced classification problems.

---

**Project Category:** Machine Learning | Binary Classification | Customer Analytics
