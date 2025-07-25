# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from py_scripts import data_preprocessing_template as dpt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

'''The test csv is 50_Startups.csv'''

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(dpt.X_train, dpt.y_train)
print(regressor.predict(dpt.X_test))

# Predicting the Test set results
y_pred = regressor.predict(dpt.X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), dpt.y_test.reshape(len(dpt.y_test),1)),1))

# Single prediction with user input
print("\n--- Make a Single Prediction ---")
rd_spend = float(input("Enter R&D Spend: "))
admin = float(input("Enter Administration: "))
marketing = float(input("Enter Marketing Spend: "))
state = input("Enter State (New York/California/Florida): ")

# Take the first row from training data as template and modify it
user_input = dpt.X_train[0:1].copy()  # Copy first row

# Find the positions of the numeric features by checking a known training sample
# The last 3 features should be R&D, Admin, Marketing based on the preprocessing
user_input[0, -3] = rd_spend    # R&D Spend  
user_input[0, -2] = admin       # Administration
user_input[0, -1] = marketing   # Marketing Spend

# Set state encoding (first 3 features are the state one-hot encoding)
user_input[0, 0] = 1 if state == "California" else 0
user_input[0, 1] = 1 if state == "Florida" else 0  
user_input[0, 2] = 1 if state == "New York" else 0

# Make prediction (input is already scaled like training data)
prediction = regressor.predict(user_input)
print(f"Predicted Profit: ${prediction[0]:.2f}")

'''
R&D Spend: 165000
Administration: 136000
Marketing Spend: 470000
State: New York

'''