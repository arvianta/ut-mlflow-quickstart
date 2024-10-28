# UT MLflow Quickstart Challenge

## About Dataset

### Problem Statement
You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a substantial amount of credit-related information. The management aims to build an intelligent system that can classify customers into credit score brackets to minimize manual efforts.

### Task
Using a person’s credit-related information, build a machine learning model that can classify their credit score.

## Dataset Columns

| Column                   | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `Customer_ID`            | Represents a unique identification of a person                |
| `Month`                  | Represents the month of the year                              |
| `Name`                   | Represents the name of a person                               |
| `Age`                    | Represents the age of the person                              |
| `SSN`                    | Represents the social security number of a person             |
| `Occupation`             | Represents the occupation of the person                       |
| `Annual_Income`          | Represents the annual income of the person                    |
| `Monthly_Inhand_Salary`  | Represents the monthly base salary of a person                |
| `Num_Bank_Accounts`      | Represents the number of bank accounts a person holds         |
| `Num_Credit_Card`        | Represents the number of other credit cards held by a person  |
| `Interest_Rate`          | Represents the interest rate on credit card                   |
| `Num_of_Loan`            | Represents the number of loans taken from the bank            |
| `Type_of_Loan`           | Represents the types of loans taken by a person               |
| `Delay_from_due_date`    | Represents the average days delayed from the payment date     |
| `Num_of_Delayed_Payment` | Represents the average number of payments delayed             |
| `Changed_Credit_Limit`   | Represents the percentage change in credit card limit         |
| `Num_Credit_Inquiries`   | Represents the number of credit card inquiries                |
| `Credit_Mix`             | Represents the classification of the mix of credits           |
| `Outstanding_Debt`       | Represents the remaining debt to be paid (in USD)             |
| `Credit_Utilization_Ratio`| Represents the utilization ratio of credit card              |
| `Credit_History_Age`     | Represents the age of credit history of the person            |
| `Payment_of_Min_Amount`  | Represents whether only the minimum amount was paid           |
| `Total_EMI_per_month`    | Represents the monthly EMI payments (in USD)                  |
| `Amount_invested_monthly`| Represents the monthly amount invested by the customer (in USD)|
| `Payment_Behaviour`      | Represents the payment behavior of the customer               |
| `Monthly_Balance`        | Represents the monthly balance amount of the customer (in USD)|

## Challenge Objective

This challenge requires the implementation of MLflow tracking in every stage of the workflow, from data preprocessing to model training and evaluation. The winning submission will be the model that achieves the highest accuracy based on **Balanced Accuracy**.

### Required Files for Submission

To participate in the challenge, please submit the following files:

1. **data_config.yaml**: 
   - This file should contain feature names and the target (prediction) variable.
   - Use the template provided for data configuration.

```yaml
feature_columns:
  - Credit_Mix
  - Payment_of_Min_Amount
  - Payment_Behaviour
  - Occupation_Accountant
  - Occupation_Architect
  - Occupation_Developer
  - Occupation_Doctor
  - Occupation_Engineer
  - Occupation_Entrepreneur
  - Occupation_Journalist
  - Occupation_Lawyer
  - Occupation_Manager
  - Occupation_Mechanic
  - Occupation_Media_Manager
  - Occupation_Musician
  - Occupation_Scientist
  - Occupation_Teacher
  - Occupation_Unknown
  - Occupation_Writer
  - Age
  - Annual_Income
  - Monthly_Inhand_Salary
  - Num_Bank_Accounts
  - Num_Credit_Card
  - Interest_Rate
  - Num_of_Loan
  - Delay_from_due_date
  - Num_of_Delayed_Payment
  - Changed_Credit_Limit
  - Num_Credit_Inquiries
  - Outstanding_Debt
  - Credit_Utilization_Ratio
  - Credit_History_Age
  - Total_EMI_per_month
  - Amount_invested_monthly
  - Monthly_Balance
  - Credit-Builder Loan
  - Personal Loan
  - Mortgage Loan
  - Home Equity Loan
  - Debt Consolidation Loan
  - Payday Loan
  - Student Loan
  - Auto Loan

target_column: Credit_Score
```

2. **Preprocessed Test Dataset**:
   - The test dataset after preprocessing steps should be submitted.

3. **Model File**:
   - Submit the trained model in either `.h5` or `.pkl` (pickle) format.

---

Good luck with the challenge, and may the best model win!
