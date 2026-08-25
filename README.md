# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Load the dataset and separate the input features (X) and target variable (Y).
2.Initialize the weight, bias, learning rate, and number of iterations. 
3. Calculate predictions and error, then update the weight and bias using Gradient Descent to minimize MSE. ([developers.google.com][1]) 
4. Repeat until convergence and use the optimized model to make predictions and evaluate its performance.

## Program:
```import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("ex3.xls")
X = data[["R&D Spend", "Administration", "Marketing Spend"]].values #convert a DataFrame or Series into a NumPy array.
y = data["Profit"].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
m, n = X.shape          # m = samples, n = features
w = np.zeros(n)         # weights (w1, w2, w3)
b = 0.0                 # bias
alpha = 0.01            # learning rate
epochs = 1000
losses = []
for i in range(epochs):
    y_hat = np.dot(X, w) + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    dw = (2/m) * np.dot(X.T, (y_hat - y))
    db = (2/m) * np.sum(y_hat - y)
    w = w - alpha * dw
    b = b - alpha * db
print("Final Weights:", w)
print("Final Bias:", b)

# Loss vs Iterations

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations (Multiple Linear Regression)")
plt.show()
/*
Program to implement the linear regression using gradient descent.
Developed by: harish kumar.G
RegisterNumber: 212225080017 
*/
```

## Output:
<img width="1162" height="846" alt="image" src="https://github.com/user-attachments/assets/351aceb3-9ac3-48b0-b201-805aec100cfa" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
