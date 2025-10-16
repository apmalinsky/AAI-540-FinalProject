# Evaluation Script for use in CI-CD Pipeline
import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    # Create a new Booster object and load the model into it
    model = xgboost.Booster()
    model.load_model("xgboost-model")

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "rmse": {"value": rmse, "standard_deviation": std},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    print("Evaluation script finished successfully.")
        