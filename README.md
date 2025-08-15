# mental_health_research

# Comparative Analysis of Machine Learning Algorithms for Predicting Depression Risk in Students

**Authors:** Mahi Arora, Gungun Jain  
**Institution:** Indira Gandhi Delhi Technical University for Women (IGDTUW)  

## Overview

This repository presents a comparative study of multiple machine learning algorithms to predict the likelihood of depression among students using the publicly available *Depression Student Dataset* sourced from Kaggle. The primary objective is to identify the most effective predictive model and evaluate the effectiveness of hybrid bagging-based ensemble methods in enhancing prediction accuracy.

## Dataset

- **Name:** Depression Student Dataset  
- **Source:** Kaggle  
- **Description:** The dataset contains survey-based attributes capturing demographic and psychometric variables associated with mental health and depression risk in students.

## Methodology

Data preprocessing included handling of missing values, label encoding, scaling, and train-test split. Following preprocessing, several machine learning algorithms were trained individually. Ensemble learning strategies using bagging were then implemented to assess improvement in predictive capability over single estimators. The following models were evaluated:

- Random Forest  
- Naïve Bayes  
- Support Vector Machine (SVM)  
- K-Nearest Neighbours (KNN)  
- Gradient Boosting  
- Bagging-based ensembles:
  - Bagging SVC  
  - Bagging KNN  
  - Bagging Random Forest  
  - Bagging Naïve Bayes  
  - Bagging Gradient Boosting  

## Evaluation Metrics

Models were assessed using the following standard performance metrics:

| Metric          | Description                                       |
|-----------------|---------------------------------------------------|
| Accuracy        | Overall correctness of classification             |
| Precision       | Correctness of positive predictions               |
| Recall          | Ability to correctly detect positive cases        |
| F1 Score        | Harmonic mean of precision and recall             |
| ROC-AUC         | Area under the ROC curve                          |

## Results

| Model         | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------|----------|-----------|--------|----------|---------|
| Random Forest | 0.821782 | 0.830189 | 0.830189 | 0.830189 | 0.887972 |
| Naïve Bayes | 0.811881 | 0.826923 | 0.811321 | 0.819048 | 0.919811 |
| SVM | 0.762376 | 0.773585 | 0.773585 | 0.773585 | 0.884827 |
| KNN | 0.693069 | 0.729167 | 0.660377 | 0.693069 | 0.777516 |
| Gradient Boosting | 0.792079 | 0.785714 | 0.830189 | 0.807339 | 0.867925 |
| Bagging SVC | 0.752475 | 0.752778 | 0.750933 | 0.750912 | 0.750393 |
| Bagging KNN | 0.683168 | 0.682411 | 0.681407 | 0.681639 | 0.681407 |
| Bagging RF | 0.792079 | 0.792857 | 0.790094 | 0.790766 | 0.790094 |
| Bagging NB | 0.821782 | 0.823963 | 0.819379 | 0.820356 | 0.819379 |
| Bagging GB | 0.772277 | 0.774659 | 0.769261 | 0.770023 | 0.769261 |

### Key Observation

Random Forest and Naïve Bayes achieved the highest overall performance among individual algorithms, while bagging-based Naïve Bayes marginally improved precision and stability. Bagging ensembles did not consistently outperform their base learners, signalling that the benefits of hybridisation are algorithm-dependent.

## Future Work

- Investigation of feature engineering techniques to improve model interpretability  
- Incorporation of boosting-based hybrid ensembles (e.g., AdaBoost, XGBoost)  
- Evaluation on larger and more diverse mental health datasets  
- Deployment of the best-performing model as a screening tool for real-time applications

## Repository Contents

├── Mental_Health_Paper.ipynb # Jupyter Notebook containing preprocessing, modelling, and results
├── README.md # Project documentation



## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/mahiii-a/mental_health_research.git

2. Install the required Python libraries (e.g., scikit-learn, pandas, numpy, matplotlib).

3. Open the Jupyter Notebook:

3. Execute all cells sequentially to reproduce the analysis and results.
