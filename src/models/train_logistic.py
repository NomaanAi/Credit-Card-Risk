from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """
    Train Logistic Regression model.
    """
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model
