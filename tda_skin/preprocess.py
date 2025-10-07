
from skimage import io, color, filters, exposure
import numpy as np

def load_grayscale(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = exposure.equalize_adapthist(img, clip_limit=0.02)
    return img

def edge_points(img, thresh: float = 0.25, max_points: int = 3000):
    e = filters.sobel(img)
    mask = e > (e.mean() + thresh * e.std())
    rr, cc = np.nonzero(mask)
    pts = np.stack([rr, cc], axis=1).astype(float)
    if pts.shape[0] == 0:
        return np.zeros((0,2))
    h, w = img.shape
    pts[:, 0] /= h
    pts[:, 1] /= w
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(7).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts
