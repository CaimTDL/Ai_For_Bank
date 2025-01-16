
# Credit Score Prediction

This project aims to train an Artificial Intelligence (AI) model to predict the **credit score** of new customers based on historical data. The model used is a **Random Forest** classifier.

## Table of Contents

- [Objective](#objective)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Credit Score Prediction](#credit-score-prediction)
- [How to Run](#how-to-run)
- [Expected Results](#expected-results)
- [Conclusion](#conclusion)

## Objective

The objective of this project is to:
1. **Train an AI model** capable of predicting the **credit score** of new customers based on historical data.
2. **Predict the credit score** for new customers using the trained model.

This project uses **classification** techniques to predict whether a customer will have a "Good", "Standard", or "Poor" credit score based on their profile.

## Technologies Used

- **Python**: Programming language used to develop the code.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing the AI model and data preprocessing.
- **LabelEncoder**: To transform categorical variables into numerical variables for model analysis.
- **RandomForestClassifier**: A decision tree-based algorithm used to predict the credit score.

## Project Structure

The project is divided into the following steps:

1. **Data Import**: Loading and displaying customer data.
2. **Data Preprocessing**: Transforming categorical variables into numerical variables using `LabelEncoder`.
3. **Model Training**: Splitting the data into training and testing sets and training the model to predict credit scores.
4. **Credit Score Prediction**: Using the trained model to predict the credit score of new customers.

## Data Preprocessing

Before training the model, the data needs to be preprocessed to ensure all features are numeric, making them suitable for analysis by the model.

### Steps performed:
- **Data Import**: The customer data is loaded from a CSV file.
  
```python
tabela = pd.read_csv("clientes.csv")
display(tabela)
```

- **Categorical Variables Transformation**: We used `LabelEncoder` to transform variables such as "profession", "credit mix", and "payment behavior" into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

coder_profession = LabelEncoder()
tabela["profissao"] = coder_profession.fit_transform(tabela["profissao"])

coder_mix_credits = LabelEncoder()
tabela["mix_credito"] = coder_mix_credits.fit_transform(tabela["mix_credito"])

coder_payment = LabelEncoder()
tabela["comportamento_pagamento"] = coder_payment.fit_transform(tabela["comportamento_pagamento"])
```

## Model Training

After preprocessing, the next step is to train the AI model. We use the **Random Forest** algorithm, which is a supervised learning algorithm based on multiple decision trees.

### Steps performed:
1. We define the target variable (`score_credito`) as the dependent variable (`y`), and the remaining columns as independent variables (`X`).
2. We split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

y = tabela["score_credito"]
X = tabela.drop(columns=["score_credito", "id_cliente"])

x_train, x_test, y_train, y_test = train_test_split(X, y)
```

3. We train the model using the training data.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
```

4. We evaluate the model's performance using the test data.

```python
from sklearn.metrics import accuracy_score

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")
```

## Credit Score Prediction

Once the model is trained, we can use it to predict the **credit score** for new customers.

### Steps performed:
1. Load the new customers' data.
2. Perform the same transformation on categorical variables as was done for the training data.
3. Use the trained model to predict the credit score.

```python
# Load new customer data
tabela_new_client = pd.read_csv("novos_clientes.csv")

# Transform categorical variables
tabela_new_client["profissao"] = coder_profession.transform(tabela_new_client["profissao"])
tabela_new_client["mix_credito"] = coder_mix_credits.transform(tabela_new_client["mix_credito"])
tabela_new_client["comportamento_pagamento"] = coder_payment.transform(tabela_new_client["comportamento_pagamento"])

# Predict credit score for new customers
new_predict = model.predict(tabela_new_client)
display(new_predict)
```

## How to Run

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/seu-repositorio.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script to train the model and predict the credit score for new customers:
    ```bash
    python predict_credit_score.py
    ```

## Expected Results

The model should be able to predict the **credit score** for new customers accurately, with an expected accuracy of around 82%, based on the historical data provided. The prediction will be made on a scale of 3 classes: "Good", "Standard", and "Poor".

## Conclusion

This project demonstrates how machine learning algorithms can be used to predict a customer's credit score. Further improvements could be made by tuning the model or incorporating additional features. The next step would be to apply cross-validation techniques to improve the model's robustness.
