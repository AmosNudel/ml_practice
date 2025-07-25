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

def find_file(filename, search_path="."):
    """Recursively search for a file with the given filename starting from search_path."""
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {search_path}")

# Find the data file path
csv_filename = 'Data.csv'  # Change this as needed for other files
csv_path = find_file(csv_filename, search_path="..")  # Search from project root

dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X:\n", X)
print("Original y:\n", y)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("After missing data imputation X:\n", X)

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("After encoding categorical X:\n", X)
# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print("After encoding categorical y:\n", y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Scaled X_train:\n", X_train)
print("Scaled X_test:\n", X_test)

