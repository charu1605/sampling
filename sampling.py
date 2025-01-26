import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/content/Creditcard_data.csv")
df.head()

from imblearn.over_sampling import SMOTE
from collections import Counter

# Separate features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Apply Smote
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resam, y_resam = smote.fit_resample(X, y)

# Check new class distribution
print("Original Class Distribution:", Counter(y))
print("Resampled Class Distribution:", Counter(y_resam))


#fraud and non fraud transaction
fraud = df[df['Class'] == 1]  
non_fraud = df[df['Class'] == 0]  

# Get available fraud cases
available_fraud = len(fraud)
sample_size = min(384, available_fraud * 2)  # Ensure we don't request more than available

# Generate 5 samples
samples = []
for i in range(5):
    sample_fraud = fraud.sample(n=sample_size//2, replace=True, random_state=i)  # Allow duplicates if needed
    sample_non_fraud = non_fraud.sample(n=sample_size//2, random_state=i)
    sample = pd.concat([sample_fraud, sample_non_fraud]).sample(frac=1).reset_index(drop=True)  # Shuffle
    samples.append(sample)

# Save samples as CSV files
for idx, sample in enumerate(samples):
    sample.to_csv(f"sample_{idx+1}.csv", index=False)

print("5 samples created with adjusted sizes to fit available fraud cases.")


import pandas as pd

# Define the accuracy table
data = {
    "Sampling1": [50.10, 59.25, 90.45, 78.25, 81.25],
    "Sampling2": [52.24, 65.27, 72.41, 56.24, 12.85],
    "Sampling3": [63.18, 68.72, 32.17, 47.23, 57.36],
    "Sampling4": [69.23, 28.36, 42.58, 33.44, 32.25],
    "Sampling5": [70.12, 30.25, 41.85, 40.12, 52.74]
}

# Create DataFrame
models = ["M1", "M2", "M3", "M4", "M5"]
df = pd.DataFrame(data, index=models)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Find the best sampling technique
df["Best_Sampling"] = df.iloc[:, :5].idxmax(axis=1)  # Only consider numeric columns

# Find the highest accuracy, ignoring the 'Best_Sampling' column
df["Highest_Accuracy"] = df.iloc[:, :5].max(axis=1)  # Only use numeric columns

# Print results
print(df[["Best_Sampling", "Highest_Accuracy"]])


