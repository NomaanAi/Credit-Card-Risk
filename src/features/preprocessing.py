from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def get_preprocessor():
    """
    Create a preprocessing pipeline.
    
    Returns:
        ColumnTransformer: Transformer for numerical and categorical features.
    """
    # Define features
    numerical_features = [
        'limit_bal', 'age', 
        'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
    ]
    
    categorical_features = ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    
    # Numerical pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any other columns not specified
    )
    
    return preprocessor

def preprocess_data(X_train, X_test):
    """
    Fit preprocessor on training data and transform both train and test.
    
    Returns:
        X_train_processed, X_test_processed, preprocessor_object
    """
    preprocessor = get_preprocessor()
    
    # Note: ColumnTransformer returns numpy array by default
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, preprocessor
