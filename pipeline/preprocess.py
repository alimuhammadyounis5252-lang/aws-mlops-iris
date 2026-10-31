import os
import pandas as pd
from sklearn.model_selection import train_test_split

base = os.environ.get("PROCESSING_BASE", "/opt/ml/processing")
df = pd.read_csv(f"{base}/input/iris.csv")
print("Input shape:", df.shape)

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

os.makedirs(f"{base}/train", exist_ok=True)
os.makedirs(f"{base}/test", exist_ok=True)
train.to_csv(f"{base}/train/train.csv", index=False)
test.to_csv(f"{base}/test/test.csv", index=False)
print("Train:", train.shape, "| Test:", test.shape)
