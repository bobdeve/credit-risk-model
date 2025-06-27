from xverse.transformer import WOE

def apply_woe_encoding(X_train, X_test, y_train):
    """
    Apply WoE transformation on training and test sets.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Target variable for training set

    Returns:
        X_train_woe, X_test_woe (pd.DataFrame): Transformed datasets
    """
    woe = WOE()
    woe.fit(X_train, y_train)
    X_train_woe = woe.transform(X_train)
    X_test_woe = woe.transform(X_test)
    
    return X_train_woe, X_test_woe
