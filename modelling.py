import pandas as pd

# Load data for a single day
data = pd.read_csv("Day3/HRL_combined.csv")

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

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load your floorsheet data
df = data  # Update with your file path

# Ensure 'Transact_No' is unique and sort by transaction number (if necessary)
df.sort_values('Transact_No', inplace=True)

# Define large order threshold (e.g., 90th percentile)
large_order_threshold = df['Quantity'].quantile(0.90)
large_orders = df[df['Quantity'] >= large_order_threshold]

# Function to calculate price change before and after a large order
def calculate_price_change(large_order, df, time_window):
    order_index = large_order.name  # Get the index of the large order

    # Calculate price before and after the order
    price_before = df.loc[(df.index < order_index) & 
                          (df.index >= order_index - time_window), 'Rate'].mean()
    price_after = df.loc[(df.index >= order_index) & 
                         (df.index < order_index + time_window), 'Rate'].mean()

    return price_after - price_before

# Apply the function to calculate price impact for each large order
large_orders['Price_Impact'] = large_orders.apply(lambda row: calculate_price_change(row, df, time_window=5), axis=1)

# Visualize the relationship between order size and price impact
plt.scatter(large_orders['Quantity'], large_orders['Price_Impact'])
plt.title('Price Impact vs. Large Order Size')
plt.xlabel('Order Size (Quantity)')
plt.ylabel('Price Impact')
plt.grid()
plt.show()

# Perform statistical analysis to quantify the relationship
X = large_orders[['Quantity']]
y = large_orders['Price_Impact']
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()
print(model.summary())
