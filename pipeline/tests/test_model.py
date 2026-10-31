import json, pathlib, importlib.util
import joblib, numpy as np, pandas as pd

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
ROOT = pathlib.Path(__file__).resolve().parents[1]

def model(workdir):
    return joblib.load(workdir / "model_out" / "model.joblib")

# --- Quality ---
def test_accuracy_above_threshold(workdir):
    report = json.loads((workdir / "evaluation" / "evaluation.json").read_text())
    assert report["metrics"]["accuracy"]["value"] >= 0.90

# --- Behaviour ---
def test_classic_setosa_is_predicted_correctly(workdir):
    X = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
    assert model(workdir).predict(X)[0] == 0

# --- Invariance ---
def test_tiny_noise_does_not_change_predictions(workdir):
    test = pd.read_csv(workdir / "test" / "test.csv")[FEATURES]
    noisy = test + np.random.default_rng(0).normal(0, 0.01, test.shape)
    m = model(workdir)
    agreement = (m.predict(test) == m.predict(noisy)).mean()
    assert agreement >= 0.95

# --- Contract (yaad hai 1D vs 2D wala 500 error?) ---
def test_single_row_input_returns_single_prediction(workdir):
    X = pd.DataFrame([[6.9, 3.1, 5.4, 2.1]], columns=FEATURES)
    assert model(workdir).predict(X).shape == (1,)

# --- Packaging (deploy ke liye zaroori) ---
def test_model_is_packaged_with_inference_code(workdir):
    assert (workdir / "model_out" / "model.joblib").exists()
    assert (workdir / "model_out" / "code" / "inference.py").exists()

def test_inference_model_fn_loads_model(workdir):
    spec = importlib.util.spec_from_file_location("inference", workdir / "model_out" / "code" / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.model_fn(str(workdir / "model_out")), "predict")

# --- Dependencies (yaad hai 1.5.0 vs 1.4.2?) ---
def test_sklearn_version_is_pinned():
    reqs = (ROOT / "train_src" / "requirements.txt").read_text()
    assert "scikit-learn==" in reqs, "sklearn ka version pin karo, warna train/serve skew!"
