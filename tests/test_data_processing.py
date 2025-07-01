import os
import sys
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

# Add project root to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_and_split_data
from src.preprocessing import get_preprocessor

input_path = 'credit-risk-model/data/raw/test.csv'
df = pd.read_csv(input_path)
def test_load_and_split_data():
    # Prepare dummy CSV file
    test_csv = input_path
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': ['A', 'B', 'A', 'B', 'C'],
        'FraudResult': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(test_csv, index=False)

    X_train, X_test, y_train, y_test = load_and_split_data(test_csv)

    # Check train-test split sizes
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

    # Check that target column is separated correctly
    assert 'FraudResult' not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})

    os.remove(test_csv)  # Clean up

def test_get_preprocessor():
    categorical_cols = ['cat_col']
    numerical_cols = ['num_col']

    preprocessor = get_preprocessor(categorical_cols, numerical_cols)

    # Should return a sklearn Pipeline or ColumnTransformer
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)

    # Check if transformers are set correctly
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert 'num' in transformer_names
    assert 'cat' in transformer_names

    # Check if pipeline steps inside numeric and categorical pipelines exist
    numeric_pipeline = preprocessor.named_transformers_['num']
    cat_pipeline = preprocessor.named_transformers_['cat']

    assert isinstance(numeric_pipeline, Pipeline)
    assert isinstance(cat_pipeline, Pipeline)
