from datetime import datetime

def calculate_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm
