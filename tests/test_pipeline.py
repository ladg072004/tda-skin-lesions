
import numpy as np
from skimage.draw import disk
from tda_skin.preprocess import edge_points
from tda_skin.topology import compute_diagrams, topo_features

def synthetic_circle(n=128, r=30):
    img = np.zeros((n,n), float)
    rr, cc = disk((n//2, n//2), r, shape=img.shape)
    img[rr, cc] = 1.0
    return img

def test_topology_features():
    img = synthetic_circle()
    pts = edge_points(img, thresh=0.0)
    res = compute_diagrams(pts, maxdim=1)
    feats = topo_features(res["dgms"])
    assert feats.shape[0] == 8
