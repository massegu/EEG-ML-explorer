# 🧠 EEG-ML Explorer

Aplicación web interactiva para **análisis, visualización y modelado de señales EEG**, orientada a exploración científica, docencia y prototipado rápido de pipelines de neurociencia computacional.

Desarrollada en **Python + Streamlit**, basada en **MNE-Python** y librerías estándar de análisis de señales y *machine learning*.

---

## 🎯 ¿Para qué sirve esta app?

EEG-ML Explorer permite:

- Explorar señales EEG en formato estándar
- Extraer *features* espectrales y espaciales
- Visualizar topomaps y componentes
- Comparar condiciones experimentales
- Entrenar modelos de *machine learning* por ventanas temporales
- Analizar **traveling waves** (ondas viajeras corticales)
- Exportar visualizaciones y animaciones (GIF)

Está pensada como **herramienta exploratoria**, no como pipeline clínico cerrado.

---

## 📂 Formatos de archivo compatibles

- **EDF / EDF+** (`.edf`)  
  (vía `mne.io.read_raw_edf`)

Opcionalmente:
- CSV de coordenadas personalizadas para montajes EEG

---

## 🧰 Tecnologías y dependencias

- **Python 3.9+**
- **Streamlit**
- **MNE-Python**
- NumPy, SciPy
- scikit-learn
- matplotlib
- imageio (para exportar GIFs)

---

## 🖥️ Plataformas compatibles

- macOS
- Linux
- Windows

Probado con:
- MNE-Python ≥ 1.5
- Streamlit ≥ 1.30

---

## 🚀 Instalación y ejecución

```bash
# Clonar el repositorio
git clone https://github.com/massegu/EEG-ML-explorer.git
cd EEG-ML-explorer

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app.py


## 🧩 Funcionalidades principales
# 1️⃣ Carga y visualización EEG
Carga de archivos EDF

Inspección básica de señales

Selección de canales

Montaje estándar (10–20 / 10–05) o personalizado

# 2️⃣ Features espectrales y espaciales
Cálculo de PSD (Welch)

Bandpower por canal y banda

Escala lineal o log10

Topomaps con:

escala lineal

log10

z-score espacial

Normalización y comparación visual

# 3️⃣ PCA sobre canales
PCA aplicado a la matriz EEG (canales como variables)

Visualización de componentes temporales

Explained variance ratio

PCA no equivale a canales: los componentes son combinaciones espaciales de canales

# 4️⃣ Comparación de condiciones (A vs B)
Definición manual de intervalos temporales

Comparación topográfica entre condiciones

Visualización:

A

B

A − B (colormap divergente)

# 5️⃣ Machine Learning por ventanas temporales
Segmentación en ventanas deslizantes

Features:

Bandpower canal × banda

Traveling wave metrics

Spatial features (AP, LR, GFP, centro de masa)

Tareas:

Clasificación

Regresión

Validación cruzada automática

Selección del mejor modelo

# 6️⃣ Traveling Waves (ondas viajeras)
Estimación de:

Dirección (θ)

Magnitud espacial |k|

Speed proxy

Visualización interactiva:

Flecha de propagación

Topomap de fase

Series temporales

Animación con Play

Exportación a GIF

Visualizaciones debug:

cos(θ), sin(θ) vs tiempo

## Diagrama Pipeline

EEG (EDF)
   │
   ▼
Preprocesado básico (MNE)
   │
   ▼
Ventanas temporales
   │
   ├── Bandpower (PSD)
   │       ├── Topomaps
   │       ├── PCA
   │       └── Spatial features
   │
   ├── Traveling waves
   │       ├── Dirección
   │       ├── |k|
   │       └── Speed
   │
   ▼
Feature matrix (X)
   │
   ▼
Machine Learning
   │
   ▼
Evaluación / Visualización


## ⚠️ Limitaciones y buenas prácticas

# Limitaciones

No incluye limpieza automática de artefactos (ICA, ASR)

Las métricas de traveling waves son aproximaciones espaciales

No sustituye análisis estadístico formal entre sujetos

Dependiente de calidad del montaje y cobertura espacial

# Buenas prácticas

Usar ventanas suficientemente largas (≥2–4 s)

Interpretar z-score solo como patrón relativo

Verificar siempre las etiquetas (debug de ventanas)

Comparar sujetos/condiciones antes de usar ML

Usar ML como herramienta exploratoria, no confirmatoria
