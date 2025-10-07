
import numpy as np
try:
    from persim import PersistenceImager
    _HAS_PI = True
except Exception:
    _HAS_PI = False

def persistence_image_vector(dgms, pixels=20):
    if not _HAS_PI:
        return np.zeros(pixels*pixels*2, dtype=float)
    pi = PersistenceImager(pixel_size=1.0/pixels)
    vecs = []
    for idx in (0,1):
        D = dgms[idx] if len(dgms) > idx else np.zeros((0,2))
        if D.size == 0:
            vecs.append(np.zeros((pixels, pixels)))
        else:
            finite = np.isfinite(D[:,1])
            Df = D[finite]
            if Df.size == 0:
                vecs.append(np.zeros((pixels, pixels)))
            else:
                img = pi.transform(Df)
                vecs.append(img)
    V = np.concatenate([v.ravel() for v in vecs])
    return V
