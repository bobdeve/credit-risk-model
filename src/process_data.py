import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 👇 Load your dataset
df = pd.read_csv('../data/processed/cleaned_data_with_rfm.csv', parse_dates=['TransactionStartTime'])

# 🎯 Define your target and drop unused columns
target = 'FraudResult'

drop_cols = ['CustomerId', 'TransactionId', 'BatchId', 'ProductId', 'TransactionStartTime']
X = df.drop(columns=[target] + drop_cols)
y = df[target]

# 🏷️ Define categorical and numerical columns
categorical_cols = [
    'CurrencyCode', 'CountryCode', 'ProviderId',
    'ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId'
]

numerical_cols = [
    'Amount', 'Value', 'Hour', 'DayOfWeek', 'TransactionDay',
    'TransactionMonth', 'TransactionYear', 'TotalAmount', 'AvgAmount',
    'TransactionCount', 'StdAmount', 'Monetary', 'Frequency', 'Recency'
]

# 🔧 Categorical transformer: fill missing, then encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# 🔧 Numerical transformer: fill missing, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 🧩 Combine transformers by column type
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numerical_transformer, numerical_cols)
])

# 🤖 Build a full pipeline with model (optional)
model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=500))
])

# 📊 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fit model (training pipeline with preprocessing)
model_pipeline.fit(X_train, y_train)

# 💾 OPTIONAL: Save transformed features to CSV
# Step 1: Transform the full dataset (X only)
X_transformed = preprocessor.fit_transform(X)

# Step 2: Get encoded feature names from OneHotEncoder
encoded_cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
all_feature_names = list(encoded_cat_cols) + numerical_cols

# Step 3: Create DataFrame and save
transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
transformed_df['FraudResult'] = y.values  # Add target column back

# Step 4: Save to file
os.makedirs('../data/processed', exist_ok=True)
transformed_df.to_csv('../data/processed/transformed_model_data.csv', index=False)

print("✅ Preprocessing complete and data saved to transformed_model_data.csv")
