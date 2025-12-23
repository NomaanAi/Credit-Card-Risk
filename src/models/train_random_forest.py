from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
