
import numpy as np
from ripser import ripser
from typing import Dict, Any

def compute_diagrams(points: np.ndarray, maxdim: int = 1) -> Dict[str, Any]:
    if points.size == 0:
        return {'dgms': [np.zeros((0,2)), np.zeros((0,2))], 'num_edges': 0}
    res = ripser(points, maxdim=maxdim)
    return res

def lifetimes(diagram: np.ndarray):
    if diagram.size == 0:
        return np.array([])
    finite = np.isfinite(diagram[:,1])
    d = diagram[finite]
    return d[:,1] - d[:,0]

def topo_features(dgms):
    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0,2))
    lt0 = lifetimes(h0)
    lt1 = lifetimes(h1)
    def fstats(lt):
        if lt.size == 0:
            return [0, 0.0, 0.0, 0.0]
        return [lt.size, float(np.sum(lt)), float(np.max(lt)), float(np.mean(lt))]
    return np.array(fstats(lt0) + fstats(lt1), dtype=float)
