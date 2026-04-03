import json
import os

import joblib
import pandas as pd


def init():
    # Azure ML sets AZUREML_MODEL_DIR to the folder that contains the registered model.
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")
    global model
    model = joblib.load(model_path)


def run(raw_data):
    """
    Expected request formats (either):
    1) {"input_data": [{...row1...}, {...row2...}]}
    2) {"input_data": {...single row...}}
    Returns: {"predictions": [0/1,...]}
    """
    try:
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8")

        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        input_data = payload.get("input_data", payload)

        rows = input_data if isinstance(input_data, list) else [input_data]
        df = pd.DataFrame(rows)

        # Must match training-time engineered feature.
        df["temp_delta_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]

        preds = model.predict(df).astype(int).tolist()
        return {"predictions": preds}
    except Exception as e:
        return {"error": str(e)}

