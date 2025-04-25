import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(X)
df['target'] = y

# Split into train and test
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Save to CSV files
train_df.to_csv('data/sample_Train_Data.csv', index=False)
test_df.to_csv('data/sample_Test_Data.csv', index=False)

print("Sample data created successfully!")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}") 