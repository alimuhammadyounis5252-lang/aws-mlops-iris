import pandas as pd

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

def load(workdir, split):
    return pd.read_csv(workdir / split / f"{split}.csv")

def test_columns_are_correct(workdir):
    for split in ["train", "test"]:
        assert list(load(workdir, split).columns) == FEATURES + ["target"]

def test_no_missing_values(workdir):
    for split in ["train", "test"]:
        assert load(workdir, split).isnull().sum().sum() == 0

def test_feature_values_in_valid_range(workdir):
    df = load(workdir, "train")
    assert (df[FEATURES] > 0).all().all(), "negative ya zero measurement mila"
    assert (df[FEATURES] < 10).all().all(), "10 cm se bara phool? shayad mm mein data aa gaya"

def test_split_sizes(workdir):
    assert len(load(workdir, "train")) == 120
    assert len(load(workdir, "test")) == 30

def test_test_set_is_stratified(workdir):
    counts = load(workdir, "test")["target"].value_counts()
    assert sorted(counts.tolist()) == [10, 10, 10]

def test_no_leakage_between_train_and_test(workdir):
    raw = pd.read_csv(workdir / "input" / "iris.csv")
    overlap = len(load(workdir, "train").merge(load(workdir, "test")))
    assert overlap <= raw.duplicated().sum(), "test rows training mein bhi hain!"
