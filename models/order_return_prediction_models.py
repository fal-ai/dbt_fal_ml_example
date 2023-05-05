import uuid
import pandas as pd
import datetime


def convert_dict_to_str(val):
    if isinstance(val, dict):
        return str(val)
    return val


def train_ml_model(features: pd.DataFrame, labels: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import pickle

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=123)

    print("Model init")
    lr_model = LogisticRegression(random_state=123)

    print("Model fitting")
    lr_model.fit(X_train, y_train)

    # Test model
    y_pred = lr_model.predict(X_test)

    print("Preparing the classification report")
    # Create a report and put it in a DataFrame
    model_name = str(uuid.uuid4())
    y_test = y_test.astype(float)
    report = classification_report(y_test, y_pred, output_dict=True)
    report["model_name"] = model_name
    report["date"] = datetime.datetime.now()
    output_df = pd.DataFrame([report])
    output_df = output_df.rename(columns={"0.0": "target_0", "1.0": "target_1"})
    output_df.set_index("model_name")

    output_df = output_df.applymap(convert_dict_to_str)

    print("Saving the model")
    # Save model weights
    with open(f"ml_models/{model_name}.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    return output_df


def model(dbt, fal):
    dbt.config(materialized="table")
    orders_df = dbt.ref("customer_orders_labeled")
    X = orders_df[['age', 'total_price']]
    y = orders_df['return']
    result = train_ml_model(X, y)
    return result
