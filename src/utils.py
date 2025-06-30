import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(path, target_col='is_high_risk'):
    df = pd.read_csv(path, parse_dates=['TransactionStartTime'])

    drop_cols = ['CustomerId', 'TransactionId', 'BatchId', 'ProductId', 'TransactionStartTime']
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)
