import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to sys.path to enable import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.proxy_target import calculate_rfm, create_proxy_target, merge_proxy_target

# 📂 Load your cleaned transaction-level data
input_path = 'credit-risk-model/data/processed/processed_transactions.csv'
df = pd.read_csv(input_path, parse_dates=['TransactionStartTime'])

# 📅 Define snapshot date
snapshot_date = df['TransactionStartTime'].max()

# 📊 Calculate RFM metrics
rfm = calculate_rfm(df, snapshot_date)

# 🤖 Cluster and generate 'is_high_risk' label
rfm_with_target = create_proxy_target(rfm)

# 🔗 Merge only 'is_high_risk' into original df
df = merge_proxy_target(df, rfm_with_target)

# 💾 Save the updated dataset with 'is_high_risk' column
output_path = 'credit-risk-model/data/processed/cleaned_and_processed_risk_rfm.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("✅ Final file saved with only 'is_high_risk':", output_path)
