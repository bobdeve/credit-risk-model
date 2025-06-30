import pytest
from src.utils import load_and_split_data

def test_load_and_split_data():
    X_train, X_test, y_train, y_test = load_and_split_data('data/processed/cleaned_processed_risk.csv')
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
