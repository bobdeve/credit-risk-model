import pandas as pd

# -----------------------------
# Aggregate Features
# -----------------------------

def generate_aggregate_features(df):
    """
    Creates aggregated transaction statistics for each customer.
    
    Features generated:
    - Total transaction amount
    - Average transaction amount
    - Count of transactions
    - Standard deviation of transaction amounts
    
    Parameters:
        df (DataFrame): Raw transaction-level data

    Returns:
        DataFrame: Aggregated features per customer
    """
    agg_df = df.groupby('CustomerId').agg({
        'Value': ['sum', 'mean', 'count', 'std']
    }).reset_index()

    # Flatten multi-level columns
    agg_df.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']
    return agg_df


# -----------------------------
# Time-based Features
# -----------------------------

def extract_time_features(df):
    """
    Extracts useful features from the transaction timestamp:
    - Hour of transaction
    - Day of the month
    - Month
    - Year

    Parameters:
        df (DataFrame): Raw data with 'TransactionStartTime' column

    Returns:
        DataFrame: Data with added time-based features
    """
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df


# -----------------------------
# RFM Features
# -----------------------------

def generate_rfm_features(df):
    """
    Computes Recency, Frequency, and Monetary (RFM) features per customer.
    
    - Recency: Days since the customer's last transaction
    - Frequency: Total number of transactions
    - Monetary: Total value spent by the customer
    
    Parameters:
        df (DataFrame): Transaction-level data

    Returns:
        DataFrame: Customer-level RFM features
    """
    # Use the latest transaction date in the dataset as reference
    recent_time = df['TransactionStartTime'].max()

    # Total monetary value per customer
    monetary = df.groupby('CustomerId')['Value'].sum().reset_index()
    monetary.columns = ['CustomerId', 'Monetary']

    # Total number of transactions per customer
    frequency = df.groupby('CustomerId')['TransactionId'].count().reset_index()
    frequency.columns = ['CustomerId', 'Frequency']

    # Days since last transaction
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency['Recency'] = (recent_time - recency['TransactionStartTime']).dt.days
    recency = recency[['CustomerId', 'Recency']]

    # Merge all RFM features
    rfm = monetary.merge(frequency, on='CustomerId').merge(recency, on='CustomerId')
    return rfm



