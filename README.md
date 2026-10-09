# 💳 Credit Card Approval Prediction Using Machine Learning

## 📌 Project Overview

The Credit Card Approval Prediction project uses machine learning classification algorithms to analyze applicant information and credit history data. The objective is to explore patterns in financial data and investigate how classification models can distinguish between different credit-related categories.

The project involves loading and exploring datasets, handling missing values, combining records, preparing features, training classification models, and evaluating their performance.

Two machine learning algorithms, Logistic Regression and Random Forest, are used to compare classification results.

## 🎯 Project Objectives

* Analyze applicant information and credit history datasets.
* Explore and preprocess financial data.
* Handle missing values and prepare relevant features.
* Combine datasets using appropriate identifiers.
* Train Logistic Regression and Random Forest classifiers.
* Evaluate and compare model performance.
* Understand class imbalance and challenges in financial classification.

## 📊 Dataset

The project uses two datasets:

### 1. Application Record — `application_record.csv`

Contains applicant-related information used for analyzing financial and demographic characteristics.

* Original dataset size: 438,557 rows and 18 columns.
* Contains applicant features that require preprocessing before modeling.
* Includes missing values in the `OCCUPATION_TYPE` column.

### 2. Credit Record — `credit_record.csv`

Contains historical credit information associated with applicants.

* Original dataset size: 1,048,575 rows and 3 columns.
* Provides credit-history records for analysis and dataset preparation.

### Data Preparation

During the project, the datasets were processed and combined, resulting in a merged dataset of approximately **36,457 rows and 20 columns**.

The original application dataset contained 134,203 missing values in the `OCCUPATION_TYPE` column.

The final dataset depends on the record-selection, merging, preprocessing, and target-label definition used in the implementation.

## 🛠️ Technologies Used

| Technology          | Purpose                                  |
| ------------------- | ---------------------------------------- |
| Python              | Programming and implementation           |
| Pandas              | Data cleaning, manipulation, and merging |
| NumPy               | Numerical operations                     |
| Scikit-learn        | Machine learning and model evaluation    |
| Logistic Regression | Baseline classification                  |
| Random Forest       | Tree-based classification                |

## ⚙️ Machine Learning Workflow

### 1. Data Loading

Loaded the application and credit history datasets from CSV files using Pandas.

### 2. Exploratory Data Analysis

Examined dataset dimensions, columns, data types, missing values, and the structure of the available records.

### 3. Data Cleaning

Identified missing values and prepared the datasets for further analysis.

### 4. Dataset Integration

Combined application records with credit history using the relevant applicant identifier and examined the resulting dataset.

### 5. Feature Preprocessing

Prepared input features for classification, including handling categorical and numerical data as required by the implementation.

### 6. Model Training

Trained two supervised machine learning algorithms:

* Logistic Regression
* Random Forest Classifier

### 7. Model Evaluation

Compared the reported classification accuracy of both models to understand their performance on the prepared dataset.

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification. It estimates the probability that an observation belongs to a particular class.

It serves as a baseline model for comparing classification performance.

### 2. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a classification result.

It can capture nonlinear relationships and interactions between input features.

Both algorithms were trained to compare their reported performance on the prepared dataset.

## 📈 Model Performance

The project produced the following reported results:

| Model                                  | Reported Accuracy |
| -------------------------------------- | ----------------: |
| Logistic Regression — later evaluation |            65.96% |
| Random Forest Classifier               |            98.49% |

An earlier Logistic Regression evaluation achieved approximately 99.34% accuracy by predicting the majority class, illustrating the limitations of accuracy on imbalanced datasets.

### Performance Analysis

**Logistic Regression — 65.96%**

The later evaluation reported an accuracy of approximately 65.96%. Further analysis is needed to understand the model's ability to identify each class correctly.

**Random Forest — 98.49%**

The Random Forest model achieved approximately 98.49% reported accuracy. However, high accuracy alone does not establish that the model generalizes well to unseen data.

Potential factors such as class imbalance, data leakage, target-label construction, and train-test splitting should be investigated before interpreting this result.

Accuracy should be complemented with precision, recall, F1-score, and a confusion matrix.

*These figures are the previously reported project results and should be verified against the actual evaluation code and output.*

## 💡 Key Learnings

* Working with large structured datasets.
* Exploring and cleaning financial data.
* Handling missing values.
* Merging datasets using identifiers.
* Preparing features for supervised learning.
* Implementing Logistic Regression and Random Forest.
* Comparing classification algorithms.
* Understanding class imbalance and evaluation metrics.
* Recognizing the importance of data quality and leakage prevention.

## 🔍 Potential Applications

Credit-related classification can support research into:

* Applicant risk assessment.
* Credit history analysis.
* Financial data analytics.
* Identification of patterns in historical credit records.
* Decision-support systems, subject to appropriate validation and oversight.

The project is intended for learning and experimentation. It should not be used to make actual credit approval decisions without rigorous validation, fairness assessments, and appropriate governance.

## 🚀 Future Improvements

* Define and document the target variable clearly.
* Improve missing-value handling and feature engineering.
* Investigate class imbalance.
* Check for data leakage and overfitting.
* Use stratified train-test splitting where appropriate.
* Evaluate precision, recall, F1-score, and confusion matrices.
* Apply cross-validation and hyperparameter tuning.
* Assess fairness across relevant applicant groups.
* Build a reproducible preprocessing and modeling pipeline.
* Test model performance on genuinely unseen data.

## 📂 Project Structure

```text
Credit-Card-Approval-Prediction/
│
├── application_record.csv
├── credit_record.csv
├── credit_card_approval.py
└── README.md
```

Adjust the filenames and folder structure to match the actual project files.

Avoid committing datasets containing personal or sensitive financial information unless their use and redistribution are permitted.

## ▶️ Getting Started

### Prerequisites

Python must be installed on your system.

### 1. Install Dependencies

```bash
python -m pip install pandas numpy scikit-learn
```

### 2. Run the Project

```bash
python credit_card_approval.py
```

Replace `credit_card_approval.py` with the actual name of your Python script if it differs.

## ✅ Conclusion

This project demonstrates the application of supervised machine learning to financial classification using applicant information and credit history data. By implementing Logistic Regression and Random Forest, it explores model training, data preprocessing, and performance evaluation while highlighting the importance of reliable labels, class balance, and proper validation.
