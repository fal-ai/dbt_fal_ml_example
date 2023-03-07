import pickle
def model(dbt, fal):
    dbt.config(materialized="table")
    models_df = dbt.ref("order_return_prediction_models")
    best_model_name = models_df[models_df.accuracy == models_df.accuracy.max()].model_name[0]
    with open(f"ml_models/{best_model_name}.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    orders_new_df = dbt.ref("customer_orders")
    predictions = loaded_model.predict(orders_new_df[["age", "total_price"]])
    orders_new_df["predicted_return"] = predictions
    return orders_new_df
