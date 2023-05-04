import pickle
import pandas as pd


def make_predictions(models_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    best_model_name = models_df[models_df.accuracy == models_df.accuracy.max()].model_name[0]
    with open(f"ml_models/{best_model_name}.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    predictions = loaded_model.predict(orders_df[["age", "total_price"]])
    orders_df["predicted_return"] = predictions
    return orders_df

def model(dbt, fal):
    dbt.config(materialized="table")
    models_df = dbt.ref("order_return_prediction_models")
    orders_df = dbt.ref("customer_orders")
    return make_predictions(models_df, orders_df)
