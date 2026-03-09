# Telco Customer Churn Analysis with AI Agent

## Overview
This project demonstrates a simple **customer churn analysis system** using a telecom dataset. The goal is to identify **high-risk customers who are likely to leave the service** and allow an **AI agent to answer questions about the dataset and model insights**.

The system combines:
- Data preprocessing
- Machine learning (Logistic Regression)
- AI-based question answering

This project was created as part of a **Data Science / AI internship exercise**.

---

## Dataset

The dataset used is the **Telco Customer Churn dataset**.

Each row represents a telecom customer and includes information such as:

- Contract type  
- Tenure (how long the customer has stayed)  
- Monthly charges  
- Payment method  
- Internet service  
- Technical support availability  
- Churn status  

### Target Variable

```
Churn
1 → Customer left the service
0 → Customer stayed
```

This makes the problem a **binary classification task**.

---

## Project Workflow

```
Dataset
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Train Logistic Regression Model
   ↓
Model Evaluation
   ↓
Feature Importance Extraction
   ↓
AI Agent Explains Dataset Insights
```

---

## Data Preprocessing

The following preprocessing steps are applied:

1. Remove unnecessary columns
2. Convert `TotalCharges` to numeric values
3. Replace missing values using the median
4. Convert `Churn` into binary format
5. Encode categorical variables using `LabelEncoder`

Example preprocessing:

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
```

---

## Machine Learning Model

A **Logistic Regression model** is used to predict churn probability.

Steps performed:

1. Split dataset into training and testing data
2. Train the model
3. Evaluate accuracy
4. Extract feature importance

Example:

```python
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
```

---

## Feature Importance

The model identifies features that influence churn the most.

Example churn drivers:

- Contract type
- Tenure
- Monthly charges
- Technical support
- Internet service type

These factors help identify **high-risk customers**.

---

## AI Agent

An AI agent is integrated using an LLM API.

The agent receives:

- Dataset summary
- Model accuracy
- Feature importance
- Statistical churn insights

Users can ask questions such as:

```
Describe the dataset
What are high-risk customer characteristics?
Which payment method has the highest churn?
```

Example interaction:

```
User: What are high-risk customers?

Agent: Customers with month-to-month contracts, shorter tenure, and higher monthly charges show a higher probability of churn.
```

---

## How to Run the Project

### 1 Install dependencies

```
pip install openai pandas scikit-learn
```

### 2 Add API Key

Set your OpenRouter API key:

```
OPENROUTER_API = your_api_key_here
```

### 3 Run the notebook or Python script

Execute the notebook or script from start to finish.

### 4 Ask questions

Example:

```python
ask("Describe the dataset")
ask("What are high-risk customer characteristics?")
ask("Which payment method has the highest churn?")
```

---

## Example Output

```
Model Accuracy: 0.81

Top churn drivers:
Contract
Tenure
MonthlyCharges
TechSupport
InternetService
```

---

## Purpose of the Model

The machine learning model learns patterns from historical customer data to estimate churn probability.

This helps identify **customers who are most likely to leave**, allowing companies to take actions to improve retention.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- OpenRouter LLM API

---

## Future Improvements

Possible improvements include:

- Using advanced models such as **XGBoost**
- Adding **SHAP explainability**
- Building a **web interface**
- Allowing the AI agent to query the dataset dynamically
