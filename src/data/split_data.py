from sklearn.model_selection import train_test_split
import pandas as pd

def split_train_test(df, target_col='target', test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): The dataframe containing features and target.
        target_col (str): Name of the target column.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
