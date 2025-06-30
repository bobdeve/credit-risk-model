import os
import sys
import pandas as pd
print("Current working directory:", os.getcwd())
df = pd.read_csv('credit-risk-model/data/processed/cleaned_data_with_rfm.csv', parse_dates=['TransactionStartTime'])
