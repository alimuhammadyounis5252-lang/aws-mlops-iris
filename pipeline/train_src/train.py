import argparse, os, shutil, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_estimators", type=int, default=87)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR"))
    a = p.parse_args()

    df = pd.read_csv(os.path.join(a.train, "train.csv"))
    print("Train shape:", df.shape)
    X, y = df.drop(columns=["target"]), df["target"]

    model = RandomForestClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth, random_state=42)
    model.fit(X, y)
    joblib.dump(model, os.path.join(a.model_dir, "model.joblib"))

    os.makedirs(os.path.join(a.model_dir, "code"), exist_ok=True)
    shutil.copy(__file__, os.path.join(a.model_dir, "code", "inference.py"))
    print("Model saved ✅")

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))
