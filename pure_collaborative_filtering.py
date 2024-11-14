import pandas as pd  # pandas for data manipulation
import numpy as np  # numpy for sure
from scipy.sparse import coo_matrix  # for constructing sparse matrix

# lightfm
from lightfm import LightFM  # model
from lightfm.evaluation import auc_score

# timing
import time


def get_interaction_matrix(
    df,
    df_column_as_row,
    df_column_as_col,
    df_column_as_value,
    row_indexing_map,
    col_indexing_map,
):

    row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values

    return coo_matrix(
        (value, (row, col)), shape=(len(row_indexing_map), len(col_indexing_map))
    )


print("Loading datasets")

aisles = pd.read_csv("datasets/aisles.csv")
departments = pd.read_csv("datasets/departments.csv")
orders = pd.read_csv("datasets/orders.csv")
order_products__prior = pd.read_csv("datasets/order_products__prior.csv")
order_products__train = pd.read_csv("datasets/order_products__train.csv")
products = pd.read_csv("datasets/products.csv")

aisles = aisles[aisles["aisle"].apply(lambda x: x != "missing" and x != "other")]
departments = departments[
    departments["department"].apply(lambda x: x != "missing" and x != "other")
]

users = np.sort(orders["user_id"].unique())
items = products["product_name"].unique()
features = pd.concat(
    [aisles["aisle"], departments["department"]], ignore_index=True
).unique()

print("Generating mapping")

user_to_index_mapping = {}
index_to_user_mapping = {}
for user_index, user_id in enumerate(users):
    user_to_index_mapping[user_id] = user_index
    index_to_user_mapping[user_index] = user_id

item_to_index_mapping = {}
index_to_item_mapping = {}
for item_index, item_id in enumerate(items):
    item_to_index_mapping[item_id] = item_index
    index_to_item_mapping[item_index] = item_id

feature_to_index_mapping = {}
index_to_feature_mapping = {}
for feature_index, feature_id in enumerate(features):
    feature_to_index_mapping[feature_id] = feature_index
    index_to_feature_mapping[feature_index] = feature_id

print("Generating train and test datasets")

user_to_product_train_df = (
    orders[orders["eval_set"] == "prior"][["user_id", "order_id"]]
    .merge(order_products__prior[["order_id", "product_id"]])
    .merge(products[["product_id", "product_name"]])[["user_id", "product_name"]]
    .copy()
)
user_to_product_train_df["product_count"] = 1
user_to_product_rating_train = user_to_product_train_df.groupby(
    ["user_id", "product_name"], as_index=False
)["product_count"].sum()

user_to_product_test_df = (
    orders[orders["eval_set"] == "train"][["user_id", "order_id"]]
    .merge(order_products__train[["order_id", "product_id"]])
    .merge(products[["product_id", "product_name"]])[["user_id", "product_name"]]
    .copy()
)

# giving rating as the number of product purchase count (including the previous purchase in the training data)
user_to_product_test_df["product_count"] = 1
user_to_product_rating_test = user_to_product_test_df.groupby(
    ["user_id", "product_name"], as_index=False
)["product_count"].sum()

user_to_product_rating_test = user_to_product_rating_test.merge(
    user_to_product_rating_train.rename(
        columns={"product_count": "previous_product_count"}
    ),
    how="left",
).fillna(0)
user_to_product_rating_test["product_count"] = user_to_product_rating_test.apply(
    lambda x: x["previous_product_count"] + x["product_count"], axis=1
)
user_to_product_rating_test.drop(columns=["previous_product_count"], inplace=True)


aisle_weight = 1
department_weight = 1

item_feature_df = products.merge(aisles).merge(departments)[
    ["product_name", "aisle", "department"]
]

# start indexing
item_feature_df["product_name"] = item_feature_df["product_name"]
item_feature_df["aisle"] = item_feature_df["aisle"]
item_feature_df["department"] = item_feature_df["department"]

# allocate aisle and department into one column as "feature"

product_aisle_df = item_feature_df[["product_name", "aisle"]].rename(
    columns={"aisle": "feature"}
)
product_aisle_df["feature_count"] = aisle_weight  # adding weight to aisle feature
product_department_df = item_feature_df[["product_name", "department"]].rename(
    columns={"department": "feature"}
)
product_department_df["feature_count"] = (
    department_weight  # adding weight to department feature
)

# combining aisle and department into one
product_feature_df = pd.concat(
    [product_aisle_df, product_department_df], ignore_index=True
)

# saving some memory
del item_feature_df
del product_aisle_df
del product_department_df


# grouping for summing over feature_count
product_feature_df = product_feature_df.groupby(
    ["product_name", "feature"], as_index=False
)["feature_count"].sum()

del aisles
del departments
del orders
del order_products__prior
del order_products__train
del products

print("Creating interaction matrices")

# generate user_item_interaction_matrix for train data
user_to_product_interaction_train = get_interaction_matrix(
    user_to_product_rating_train,
    "user_id",
    "product_name",
    "product_count",
    user_to_index_mapping,
    item_to_index_mapping,
)

# generate user_item_interaction_matrix for test data
user_to_product_interaction_test = get_interaction_matrix(
    user_to_product_rating_test,
    "user_id",
    "product_name",
    "product_count",
    user_to_index_mapping,
    item_to_index_mapping,
)

# generate item_to_feature interaction
product_to_feature_interaction = get_interaction_matrix(
    product_feature_df,
    "product_name",
    "feature",
    "feature_count",
    item_to_index_mapping,
    feature_to_index_mapping,
)

del user_to_product_rating_train
del user_to_product_rating_test

print("Training model")

model_without_features = LightFM(loss="warp")

# fitting into user to product interaction matrix only / pure collaborative filtering factor

start = time.time()
# ===================

model_without_features.fit(
    user_to_product_interaction_train,
    user_features=None,
    item_features=None,
    sample_weight=None,
    epochs=1,
    num_threads=0,
    verbose=True,
)

# ===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# auc metric score (ranging from 0 to 1)

start = time.time()
# ===================

auc_without_features = auc_score(
    model=model_without_features,
    test_interactions=user_to_product_interaction_test,
    num_threads=4,
    check_intersections=False,
)
# ===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print(
    "average AUC without adding item-feature interaction = {0:.{1}f}".format(
        auc_without_features.mean(), 2
    )
)
