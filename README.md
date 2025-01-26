# Sampling Analysis 

## Overview
This project analyzes different sampling techniques applied to five machine learning models (M1–M5). The goal is to determine which sampling technique yields the highest accuracy for each model.

## Problem Statement
Initially, an error occurred when trying to compute the highest accuracy values. The issue was that the `Best_Sampling` column contained string values (sampling method names), causing a `TypeError` when attempting numerical operations on the dataframe.

## Solution
To resolve this issue:
1. We ensured that only numeric columns were used when calculating the highest accuracy.
2. Used `df.iloc[:, :5]` to explicitly select the first five numerical columns for computations.
3. Stored the best sampling technique separately to avoid mixing strings and numerical values.

## Files
- **sampling_analysis.py**: Python script for analyzing sampling techniques.
- **sampling_results.csv**: Output file containing the best sampling technique and highest accuracy for each model.


## Discussion
By applying the fix, we correctly identify the best sampling technique and the highest accuracy for each model without errors. The corrected approach ensures robust handling of numerical computations while preserving categorical information separately.

## Example Output
```
    Best_Sampling  Highest_Accuracy
M1    Sampling5              70.12
M2    Sampling3              68.72
M3    Sampling1              90.45
M4    Sampling1              78.25
M5    Sampling1              81.25
```

