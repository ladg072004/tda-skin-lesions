
import argparse
import numpy as np
from joblib import load
from .preprocess import load_grayscale, edge_points
from .topology import compute_diagrams, topo_features
from .features import persistence_image_vector

def build_feature(path):
    img = load_grayscale(path)
    pts = edge_points(img)
    dgms = compute_diagrams(pts, maxdim=1)['dgms']
    v_hand = topo_features(dgms)
    v_pi = persistence_image_vector(dgms, pixels=16)
    return np.concatenate([v_hand, v_pi])

def main(img_path: str, model_path: str = "models/model.joblib"):
    model = load(model_path)
    v = build_feature(img_path).reshape(1, -1)
    proba = model.predict_proba(v)[0,1]
    print(f"Score (probabilidad clase 1): {proba:.4f}")
    return proba

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, type=str)
    ap.add_argument("--model", type=str, default="models/model.joblib")
    args = ap.parse_args()
    main(args.img, args.model)
