import os, json, tarfile, joblib
import pandas as pd
from sklearn.metrics import accuracy_score

base = os.environ.get("PROCESSING_BASE", "/opt/ml/processing")
with tarfile.open(f"{base}/model/model.tar.gz") as tar:
    tar.extractall(f"{base}/model")

model = joblib.load(f"{base}/model/model.joblib")
test = pd.read_csv(f"{base}/test/test.csv")
X, y = test.drop(columns=["target"]), test["target"]

acc = accuracy_score(y, model.predict(X))
print("Test accuracy:", acc)

os.makedirs(f"{base}/evaluation", exist_ok=True)
with open(f"{base}/evaluation/evaluation.json", "w") as f:
    json.dump({"metrics": {"accuracy": {"value": acc}}}, f)
