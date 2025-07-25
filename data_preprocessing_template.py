import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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

print(X)
print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

