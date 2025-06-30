# src/proxy_target.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df, snapshot_date):
    """
    Calculate Recency, Frequency, and Monetary values for each customer.

    Args:
        df (pd.DataFrame): Raw transaction data with 'CustomerId', 'TransactionStartTime', and 'Value' columns.
        snapshot_date (pd.Timestamp): Reference date for calculating recency.

    Returns:
        pd.DataFrame: DataFrame with CustomerId and their R, F, M values.
    """
    # Recency: Days since last transaction for each customer
    recency_df = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df['TransactionStartTime']).dt.days

    # Frequency: Number of transactions per customer
    frequency_df = df.groupby('CustomerId')['TransactionId'].count().reset_index()
    frequency_df.rename(columns={'TransactionId': 'Frequency'}, inplace=True)

    # Monetary: Total transaction value per customer
    monetary_df = df.groupby('CustomerId')['Value'].sum().reset_index()
    monetary_df.rename(columns={'Value': 'Monetary'}, inplace=True)

    # Merge R, F, M
    rfm = recency_df.merge(frequency_df, on='CustomerId').merge(monetary_df, on='CustomerId')

    # Keep only relevant columns
    rfm = rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary']]
    return rfm


def create_proxy_target(rfm_df, n_clusters=3, random_state=42):
    """
    Cluster customers based on RFM and define high-risk proxy label.

    Args:
        rfm_df (pd.DataFrame): DataFrame with CustomerId, Recency, Frequency, Monetary.
        n_clusters (int): Number of clusters for KMeans.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: rfm_df with added 'Cluster' and 'is_high_risk' columns.
    """
    # 1. Scale the RFM features for meaningful clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # 2. Fit K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)

    rfm_df['Cluster'] = clusters

    # 3. Analyze clusters to find the high-risk one:
    # High risk = high Recency (inactive), low Frequency and Monetary
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })

    # The high risk cluster should have max Recency and min Frequency & Monetary
    # Find cluster with highest Recency and lowest Frequency and Monetary
    # You can use a simple scoring heuristic:
    cluster_summary['risk_score'] = cluster_summary['Recency'] - cluster_summary['Frequency'] - cluster_summary['Monetary']
    high_risk_cluster = cluster_summary['risk_score'].idxmax()

    # 4. Assign binary target: 1 for high-risk cluster, else 0
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)

    return rfm_df


def merge_proxy_target(df, rfm_with_target):
    """
    Merge the is_high_risk proxy target back into the main dataset.

    Args:
        df (pd.DataFrame): Main transaction-level dataset.
        rfm_with_target (pd.DataFrame): RFM dataframe with is_high_risk labels.

    Returns:
        pd.DataFrame: The original df with an added is_high_risk column.
    """
    # Merge on CustomerId (left join to keep all transactions)
    df = df.merge(rfm_with_target[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

    # Optional: fill missing is_high_risk with 0 (if any customers missing)
    df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)

    return df
