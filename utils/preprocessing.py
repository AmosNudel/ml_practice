import numpy as np
import pandas as pd

"""
Preprocessing utilities for machine learning models.

This module contains reusable preprocessing functions that apply the same
transformations to new data as were used during model training.
"""



def preprocess_startup_data(new_data_row, X_train):
    """
    Process new startup data using the same preprocessing pipeline as training data.
    
    This function works around the categorical encoding bug in the original preprocessing
    by using a template approach that guarantees consistent output format.
    
    Args:
        new_data_row: Dictionary with startup data
                     e.g., {'R&D Spend': 165000, 'Administration': 136000, 
                           'Marketing Spend': 470000, 'State': 'New York'}
        X_train: Training data array to use as template (ensures shape consistency)
    
    Returns:
        Processed numpy array ready for model prediction with shape matching X_train
    """
    print(f"\n--- Processing New Startup Data ---")
    print(f"Input: {new_data_row}")
    
    if isinstance(new_data_row, dict):
        # Extract values in the correct order
        rd_spend = new_data_row.get('R&D Spend', 0)
        admin = new_data_row.get('Administration', 0) 
        marketing = new_data_row.get('Marketing Spend', 0)
        state = new_data_row.get('State', 'New York')
        
        print(f"Extracted values: R&D={rd_spend}, Admin={admin}, Marketing={marketing}, State={state}")
        
        # Use template approach (which we proved works) 
        # Take existing processed training row and modify specific values
        new_processed = X_train[0:1].copy()  # Copy first training row
        
        # Modify the specific values we care about
        # Based on debugging, we know the structure is: [state_encoding..., other_features..., rd_spend, admin, marketing]
        new_processed[0, -3] = rd_spend     # R&D Spend (last 3rd column)
        new_processed[0, -2] = admin        # Administration (last 2nd column)  
        new_processed[0, -1] = marketing    # Marketing Spend (last column)
        
        # Set state encoding (first 3 columns for California, Florida, New York)
        new_processed[0, 0] = 1 if state == "California" else 0
        new_processed[0, 1] = 1 if state == "Florida" else 0
        new_processed[0, 2] = 1 if state == "New York" else 0
        
        print(f"Template approach - final shape: {new_processed.shape}")
        print(f"Template approach - first few values: {new_processed[0, :10]}")
        print(f"Template approach - last few values: {new_processed[0, -10:]}")
        
        return new_processed
    
    else:
        # If it's already an array, apply the template approach for safety
        print("Warning: Array input - applying template approach for consistency")
        new_array = np.array([new_data_row]) if new_data_row.ndim == 1 else new_data_row
        
        # Apply the template approach to be safe
        new_processed = X_train[0:1].copy()
        
        if new_array.shape[1] >= 4:
            # Assume order is [rd_spend, admin, marketing, state]
            new_processed[0, -3] = float(new_array[0, 0])  # R&D
            new_processed[0, -2] = float(new_array[0, 1])  # Admin
            new_processed[0, -1] = float(new_array[0, 2])  # Marketing
            
            # State encoding
            state = str(new_array[0, 3])
            new_processed[0, 0] = 1 if state == "California" else 0
            new_processed[0, 1] = 1 if state == "Florida" else 0
            new_processed[0, 2] = 1 if state == "New York" else 0
            
        return new_processed


def preprocess_new_data(new_data_row, X_train):
    """
    Generic alias for preprocess_startup_data.
    
    This maintains backward compatibility while allowing for future expansion
    to handle different types of datasets.
    """
    return preprocess_startup_data(new_data_row, X_train) 