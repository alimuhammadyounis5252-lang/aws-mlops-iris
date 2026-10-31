import numpy as np, pandas as pd, boto3, io

s3 = boto3.client("s3")
obj = s3.get_object(Bucket="ali886-mlops-data-2026", Key="data/raw/iris.csv")
base = pd.read_csv(io.BytesIO(obj["Body"].read())).drop(columns=["target"])

# Production data simulate karo: petals 40% bare + thora noise
rng = np.random.default_rng(0)
prod = base.sample(100, replace=True, random_state=1).reset_index(drop=True)
prod["petal length (cm)"] *= 1.4
prod["petal width (cm)"] *= 1.4
prod += rng.normal(0, 0.1, prod.shape)

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

print(f"{'Feature':<20}{'Base mean':>10}{'Prod mean':>10}{'PSI':>8}  Status")
print("-" * 62)
for col in base.columns:
    p = psi(base[col], prod[col])
    status = "OK" if p < 0.1 else ("WATCH" if p < 0.25 else "DRIFT!")
    print(f"{col:<20}{base[col].mean():>10.2f}{prod[col].mean():>10.2f}{p:>8.3f}  {status}")

# ---- PSI ko CloudWatch mein bhejo ----
cw = boto3.client("cloudwatch")
cw.put_metric_data(
    Namespace="MLOps/Iris",
    MetricData=[{"MetricName": "PSI", "Dimensions": [{"Name": "Feature", "Value": col}],
                 "Value": psi(base[col], prod[col])} for col in base.columns],
)
print("\nPSI metrics sent to CloudWatch (namespace MLOps/Iris) ✅")
