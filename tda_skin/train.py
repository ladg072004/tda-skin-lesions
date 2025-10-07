
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from joblib import dump

from .preprocess import load_grayscale, edge_points
from .topology import compute_diagrams, topo_features
from .features import persistence_image_vector

def build_feature(path):
    img = load_grayscale(path)
    pts = edge_points(img)
    dgms = compute_diagrams(pts, maxdim=1)['dgms']
    v_hand = topo_features(dgms)
    v_pi = persistence_image_vector(dgms, pixels=16)
    v = np.concatenate([v_hand, v_pi])
    return v

def main(data_dir: str = "sample_images", labels_csv: str = None, out_dir: str = "models"):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if labels_csv is None:
        labels_csv = data_dir / "labels.csv"
    df = pd.read_csv(labels_csv)
    X, y = [], []
    for _, row in df.iterrows():
        p = data_dir / row["filename"]
        X.append(build_feature(p))
        y.append(int(row["label"]))
    X = np.vstack(X); y = np.array(y)

    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=4000))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    aucs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], proba))
    metrics = {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)), "n": int(len(y))}

    pipe.fit(X, y)
    dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model to", out_dir / "model.joblib")
    print("CV AUC:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="sample_images")
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--out", type=str, default="models")
    args = ap.parse_args()
    main(args.data_dir, args.labels, args.out)
