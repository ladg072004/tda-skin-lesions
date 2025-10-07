
# TDA Skin Lesions (Demo)
**Análisis Topológico de Lesiones Cutáneas mediante Homología Persistente (Python).**  
> Propósito educativo / triage. No es diagnóstico clínico.

## Instalación
```bash
pip install -r requirements.txt
```

## Entrenamiento (demo con datos sintéticos)
```bash
python -m tda_skin.train --data_dir sample_images --out models
```

## Inferencia
```bash
python -m tda_skin.infer --img sample_images/img_00.png --model models/model.joblib
```

## Demo web (Streamlit)
```bash
streamlit run -m tda_skin.demo_app
```

## Ética y límites
- Uso educativo. No sustituye decisiones clínicas.
- Para datos reales usar ISIC/HAM10000 respetando licencias.
