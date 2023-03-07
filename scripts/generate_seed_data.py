import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate example dataset
np.random.seed(123)
n_customers = 1000
customers = pd.DataFrame({
    'customer_id': np.arange(n_customers),
    'age': np.random.randint(18, 65, size=n_customers)
})

orders = pd.DataFrame({
    'order_id': np.arange(n_customers),
    'customer_id': np.random.choice(customers['customer_id'], size=n_customers),
    'total_price': np.random.uniform(10, 1000, size=n_customers)
})

# Merge customers and orders DataFrames
merged_df = orders.merge(customers, on='customer_id', how='left')

# Determine whether or not an order will be returned based on customer age and total price
def get_return_probability(age, total_price):
    age_factor = (50 - age) / 50
    price_factor = total_price / 1000
    return_probability = age_factor * price_factor * 2
    return_probability = min(return_probability, 1)  # set return_probability to 1 if it is greater than 1
    return_probability = max(return_probability, 0)  # set return_probability to 0 if it is less than 0
    return return_probability

# Determine whether or not an order will be returned based on customer age and total price
p = merged_df.apply(lambda row: get_return_probability(row['age'], row['total_price']), axis=1)
merged_df['return'] = np.random.binomial(n=1, p=p)

merged_df.to_csv("seeds/raw_customer_order_data_labeled.csv", index=False)

new_orders = pd.DataFrame({
    'order_id': np.arange(n_customers, n_customers+50),
    'customer_id': np.random.choice(customers['customer_id'], size=50),
    'total_price': np.random.uniform(10, 1000, size=50)
})

new_merged_df = new_orders.merge(customers, on='customer_id', how='left')
new_merged_df.to_csv("seeds/raw_customer_order_data.csv", index=False)
