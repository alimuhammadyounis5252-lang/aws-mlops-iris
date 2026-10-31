import os, sys, subprocess, tarfile, pathlib, pytest
from sklearn.datasets import load_iris

ROOT = pathlib.Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    """Pipeline ko locally chalao: preprocess -> train -> evaluate."""
    base = tmp_path_factory.mktemp("proc")
    (base / "input").mkdir()
    load_iris(as_frame=True).frame.to_csv(base / "input" / "iris.csv", index=False)
    env = {**os.environ, "PROCESSING_BASE": str(base)}

    subprocess.run([sys.executable, str(ROOT / "preprocess.py")], env=env, check=True)

    model_out = base / "model_out"
    model_out.mkdir()
    subprocess.run([sys.executable, str(ROOT / "train_src" / "train.py"),
                    "--train", str(base / "train"), "--model_dir", str(model_out)], check=True)

    (base / "model").mkdir()
    with tarfile.open(base / "model" / "model.tar.gz", "w:gz") as t:
        for p in model_out.iterdir():
            t.add(p, arcname=p.name)

    subprocess.run([sys.executable, str(ROOT / "evaluate.py")], env=env, check=True)
    return base
