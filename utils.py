from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np  # Since you're using numpy methods for dtype checking
import pandas as pd

def preprocess_dataframe(df):
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False)  # Usa sparse_output=False per ottenere un array denso
    
    processed_df = df.copy()
    
    # Save name of columns before processing to avoid losing them after transformations.
    original_columns = processed_df.columns.tolist()
    
    for col in original_columns:
        # Check if the column is categorical or numerical  
        if processed_df[col].dtype in [np.float64, np.int64]:
            # Normalize numerical columns: centered around mean and normalized to unit variance. 
            # It means that daata should be centered to zero and +-1 correspond to +-std from the mean.
            processed_df[col] = scaler.fit_transform(processed_df[[col]]) 

        elif processed_df[col].dtype == object:
            unique_values = processed_df[col].unique()
            # Check if the column is binary.
            if set(unique_values) == {'Yes', 'No'}:  
                # Map Yes/No in 1/0
                processed_df[col] = processed_df[col].map({'Yes': 1.0, 'No': 0.0})
            # Check if the column is categorical with more than 2 unique values.
            else:  
                # One-Hot Encoding for cathegoriacl columns
                encoded_values = encoder.fit_transform(processed_df[[col]])
                # Create names for the new columns
                encoded_col_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_values, columns=encoded_col_names, index=df.index)
                # Add the new columns to the processed DataFrame
                processed_df = pd.concat([processed_df, encoded_df], axis=1)
                # Remove the original column
                processed_df.drop(columns=[col], inplace=True)  
    # Return the processed DataFrame (still as a dataFrame)  
    return processed_df  