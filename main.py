import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
import numpy as np

# Load data
orders = pd.read_csv('data/orders.csv')
order_products = pd.read_csv('data/order_products__train.csv')
products = pd.read_csv('data/products.csv')

# Merge and preprocess data
merged_data = pd.merge(order_products, orders, on='order_id')[['user_id', 'product_id']]
merged_data = merged_data.drop_duplicates(subset=['user_id', 'product_id'])

# Initialize LightFM dataset
dataset = Dataset()
dataset.fit(users=merged_data['user_id'].unique(), items=merged_data['product_id'].unique())
(interactions, _) = dataset.build_interactions([(x[0], x[1]) for x in merged_data.values])

# Train model
model = LightFM(loss='warp')
model.fit(interactions, epochs=30, num_threads=2)

# Evaluate model
train_precision = precision_at_k(model, interactions, k=5).mean()
print(f'Train precision at k=5: {train_precision}')

# Save recommendations for a user
def recommend_products(model, user_id, num_items=5):
    scores = model.predict(user_ids=user_id, item_ids=np.arange(interactions.shape[1]))
    top_items = np.argsort(-scores)[:num_items]
    return top_items

# Example recommendation
user_id = 0  # Adjust based on your data
print("Recommended products:", recommend_products(model, user_id))
