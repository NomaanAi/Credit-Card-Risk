from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42):
    """
    Train Gradient Boosting model.
    """
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
