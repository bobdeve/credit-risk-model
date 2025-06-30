import pandas as pd
from datetime import datetime

import sys
import os

# Add project root to sys.path to enable import from src
sys.path.append(os.path.abspath('.'))  # Adjust '.' if your notebook runs in a subfolder

from proxy_target import calculate_rfm, create_proxy_target, merge_proxy_target

# Load your main processed dataset
df = pd.read_csv('../data/processed/cleaned_data_with_rfm.csv', parse_dates=['TransactionStartTime'])

# Define snapshot date as max date in dataset (or your chosen fixed date)
snapshot_date = df['TransactionStartTime'].max()

# Step 1: Calculate RFM metrics
rfm = calculate_rfm(df, snapshot_date)

# Step 2: Cluster and create proxy target
rfm_with_target = create_proxy_target(rfm)

# Step 3: Merge proxy target back into main dataframe
df = merge_proxy_target(df, rfm_with_target)

# Now df has the 'is_high_risk' binary label for modeling
print(df[['CustomerId', 'is_high_risk']].drop_duplicates().head(50))
