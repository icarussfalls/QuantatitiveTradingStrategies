import pandas as pd

# Load data for a single day
data = pd.read_csv("Day3/NABIL_09-15-2024.csv")

# Ensure 'Quantity' and 'Rate' columns are numeric
data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
data['Rate'] = pd.to_numeric(data['Rate'], errors='coerce')

# Fill NaN values
data['Quantity'].fillna(data['Quantity'].mean(), inplace=True)
data['Rate'].fillna(data['Rate'].mean(), inplace=True)

# Aggregate features
summary_features = {
    'mean_quantity': data['Quantity'].mean(),
    'median_quantity': data['Quantity'].median(),
    'std_quantity': data['Quantity'].std(),
    'mean_rate': data['Rate'].mean(),
    'median_rate': data['Rate'].median(),
    'std_rate': data['Rate'].std(),
}

summary_df = pd.DataFrame(summary_features, index=[0])
print(summary_df)
