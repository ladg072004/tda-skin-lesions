# --- FIX para imports cuando se ejecuta con "streamlit run" ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from joblib import load
from skimage import io, color
from persim import plot_diagrams
from tda_skin.preprocess import load_grayscale, edge_points 
from tda_skin.topology import compute_diagrams, topo_features
from tda_skin.features import persistence_image_vector

st.set_page_config(page_title="TDA Skin Lesions Demo", layout="centered")
st.title("Análisis Topológico de Lesiones Cutáneas (Demo)")
st.write("**Propósito educativo / triage**. No es un dispositivo médico.")

@st.cache_resource
def load_model(path="models/model.joblib"):
    try:
        return load(path)
    except Exception:
        return None

model = load_model()

up = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg","jpeg","png"])
if up is not None:
    img = io.imread(BytesIO(up.read()))
    if img.ndim == 3:
        g = color.rgb2gray(img)
    else:
        g = img.astype(float)
    g = (g - g.min())/(g.max()-g.min()+1e-8)

    st.image(img, caption="Imagen cargada", use_column_width=True)

    pts = edge_points(g)
    res = compute_diagrams(pts, maxdim=1)
    dgms = res["dgms"]
    feats = np.concatenate([topo_features(dgms), persistence_image_vector(dgms, pixels=16)])

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    plot_diagrams(dgms, ax=ax)
    st.pyplot(fig)

    if model is not None:
        v = feats.reshape(1,-1)
        try:
            proba = model.predict_proba(v)[0,1]
            st.success(f"Puntuación (clase 1): {proba:.3f}")
        except Exception:
            st.warning("Modelo entrenado no disponible/compatible. Solo se muestran diagramas.")
    else:
        st.info("Entrena con `python -m tda_skin.train` para habilitar predicción.")
