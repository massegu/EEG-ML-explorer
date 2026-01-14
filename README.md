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

