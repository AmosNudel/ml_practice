import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# For missing data
from sklearn.impute import SimpleImputer
# For encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# For feature scaling
from sklearn.preprocessing import StandardScaler

# Import the file selection functionality
from select_data import csv_path

# Use the selected file path directly
dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X:\n", X)
print("Original y:\n", y)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if X.shape[1] > 1:
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    print("After missing data imputation X:\n", X)
else:
    print("Skipping missing data imputation: X has only", X.shape[1], "column(s)")

# Encoding categorical data only if categorical columns exist
# If X is a numpy array, convert to DataFrame for easier dtype checking
if not isinstance(X, pd.DataFrame):
    X_df = pd.DataFrame(X)
else:
    X_df = X

categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_cols:
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
    X = ct.fit_transform(X_df)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    print("After encoding categorical X:\n", X)
else:
    print("No categorical columns found. Skipping encoding.")
    # If X_df was created, convert back to numpy array
    X = X_df.values if isinstance(X_df, pd.DataFrame) else X

# Encoding the Dependent Variable only if it's categorical
if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("After encoding categorical y:\n", y)
else:
    print("Target variable is numeric. Skipping target encoding.")

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Feature Scaling
sc = StandardScaler()
if X_train.shape[1] > 3:
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])
    print("Scaled X_train:\n", X_train)
    print("Scaled X_test:\n", X_test)
else:
    print("Skipping feature scaling: dataset has", X_train.shape[1], "column(s), scaling starts from column 3")

