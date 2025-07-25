# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from py_scripts import data_preprocessing_template as dpt
from py_scripts.utils import preprocess_new_data
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

# Single prediction with user input - PROPER ML ENGINEERING APPROACH
print("\n--- Make a Single Prediction ---")
rd_spend = float(input("Enter R&D Spend: "))
admin = float(input("Enter Administration: "))
marketing = float(input("Enter Marketing Spend: "))
state = input("Enter State (New York/California/Florida): ")

# Use the preprocessing function from the training pipeline
user_input_dict = {
    'R&D Spend': rd_spend,
    'Administration': admin,
    'Marketing Spend': marketing,
    'State': state
}

# Process user input using the EXACT same pipeline as training
processed_input = preprocess_new_data(user_input_dict, dpt.X_train)

# Make prediction
if processed_input.shape[1] == dpt.X_train.shape[1]:
    prediction = regressor.predict(processed_input)
    print(f"\n🎯 Predicted Profit: ${prediction[0]:.2f}")
    print("\n✅ SUCCESS! User input processed through same pipeline as training data.")
else:
    print(f"\n❌ Shape mismatch: {processed_input.shape[1]} vs {dpt.X_train.shape[1]}")
    print("   This indicates the preprocessing function needs adjustment.")

'''
R&D Spend: 165000
Administration: 136000
Marketing Spend: 470000
State: Florida
prediction: $475033346.56

R&D Spend: 130000
Administration: 100000
Marketing Spend: 300000
State: Florida
prediction: $258439926.15

R&D Spend: 330000
Administration: 300000
Marketing Spend: 700000
State: New York
prediction: $608363197.50

'''