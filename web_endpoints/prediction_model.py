import os
from fal_serverless import isolated

ML_MODEL_ID = os.environ["ML_MODEL_ID"]
ML_MODELS_HOME = os.environ["ML_MODELS_HOME"]

@isolated(requirements=["pandas", "scikit-learn"], serve=True)
def will_return(age: float, total_price: float) -> bool:
    import pandas as pd
    import pickle
    with open(f"{ML_MODELS_HOME}/{ML_MODEL_ID}.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    df = pd.DataFrame({"age": [age], "total_price": [total_price]})
    predictions = loaded_model.predict(df[["age", "total_price"]])
    return {"prediction": int(predictions[0])}
