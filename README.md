Home Credit Default Risk
This repository contains the code for the Home Credit Default Risk competition on Kaggle. The main objective of this project is to predict whether a loan applicant will be able to repay a loan or not. This is a classic classification problem in the financial domain.
Project Overview
The project follows a structured machine learning workflow, which includes:
Data Preprocessing and Feature Engineering: Extensive feature engineering is performed on multiple data sources to create meaningful features for the predictive model. Each data source is processed in a separate, modular script.
Feature Selection: A Random Forest model is used to determine feature importance, and less important features are filtered out to reduce noise and model complexity.
Model Training and Hyperparameter Tuning: Two models, Logistic Regression and a Decision Tree Classifier, are trained on the engineered features. Hyperparameter tuning is performed using Hyperopt to find the optimal parameters for each model.
Prediction and Submission: The trained models are used to make predictions on the test set, and a submission file is generated in the format required by the Kaggle competition.
Project Structure
The project is organized into the following files and directories:
notebook_main.ipynb: The main Jupyter Notebook that orchestrates the entire workflow, from data loading and feature engineering to model training and prediction.
feature_engineering/: This directory contains Python scripts for feature engineering on different data sources.
application_fe.py: Processes the main application data (application_train.csv and application_test.csv).
bureau_fe.py: Processes the bureau data, which contains information about the applicant's previous loans with other financial institutions.
bureau_balance_fe.py: Processes the bureau balance data.
credit_card_balance_fe.py: Processes credit card balance data for previous credits.
installment_payment_fe.py: Processes installment payment history.
pos_cash_balance_fe.py: Processes point-of-sale and cash loan balance data.
previous_application_fe.py: Processes previous loan application data with Home Credit.
merge.py: Contains the function to merge all the engineered feature sets into a single training and testing dataset. It also includes the feature selection logic.
model.py: Defines the functions for training the Logistic Regression and Decision Tree models, including hyperparameter tuning with Hyperopt.
cleaned_data/: (Assumed) This directory should contain the cleaned CSV files used as input for the feature engineering scripts.
Feature Engineering
A significant part of this project is dedicated to feature engineering. Key techniques used include:
One-Hot Encoding: Categorical features are converted into a numerical format using one-hot encoding.
Domain-Specific Ratios: New features are created based on domain knowledge, such as:
CREDIT_INCOME_RATIO: The ratio of the loan amount to the applicant's income.
ANNUITY_INCOME_RATIO: The ratio of the loan annuity to the applicant's income.
DAYS_EMPLOYED / DAYS_BIRTH: The ratio of employment duration to the applicant's age.
Aggregations: For data sources with multiple records per applicant (like bureau or previous_application), features are aggregated using functions like mean, sum, min, max, and count.
Handling Missing Values: A SimpleImputer is used to fill missing values with a constant value (0) after the merging step, ensuring the models can handle the complete dataset.
Modeling and Evaluation
Feature Selection
A Random Forest Classifier is trained on the full feature set to calculate the importance of each feature.
Features with an importance score below a certain threshold (0.0001) are discarded. This helps in reducing the dimensionality of the data and improving model performance.
Model Training and Hyperparameter Tuning
Logistic Regression and Decision Tree Classifier are the two classification models implemented.
Hyperopt, a library for Bayesian optimization, is used to efficiently search for the best hyperparameters for each model. The objective function for optimization is the ROC AUC score, evaluated using 3-fold cross-validation.
SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address the class imbalance between defaulted and non-defaulted loans.
