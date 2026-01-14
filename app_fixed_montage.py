
"""
Streamlit EEG Explorer (no H2O) + montage-aware traveling waves
---------------------------------------------------------------
What this app does:
- Upload EEG data (EDF is supported out of the box via MNE)
- Preprocess (optional bandpass)
- Feature extraction (band power, PSD summary)
- PCA visualization
- Train & select best of 5 ML models (sklearn) with cross-validation
- Traveling waves analysis using electrode positions (10-20 / custom)

How to customize electrode positions (montage):
1) If your channels follow standard 10-20 labels (Fp1, Fp2, F3, F4, ...),
   just pick "standard_1020" and the app will fetch 3D positions from MNE.
2) If you use a custom montage OR a subset that isn't in the standard set,
   paste positions in the "Custom electrode positions" box as CSV lines:

   # name,x,y,z   (in meters; any consistent unit is okay, but meters is standard)
   Fp1,-0.0294,0.0836,0.0310
   Fp2, 0.0294,0.0836,0.0310
   Fz,  0.0000,0.0600,0.0750
   ...

   Tips:
   - You only need to provide positions for the channels you want to use in
     traveling wave estimation. Missing channels will be ignored (with a warning).
   - If you have X/Y only, you can set z=0.0; wave direction will still work.

Traveling waves method (lightweight, montage-aware):
- Bandpass EEG in a selected band
- Hilbert transform -> instantaneous phase per channel
- For each timepoint, fit a 2D plane to the (unwrapped) phase:
    phase ~ kx*x + ky*y + c
  where (x,y) are electrode coordinates projected to 2D.
- The estimated wave vector k = (kx, ky) gives:
    direction = atan2(ky, kx)
    spatial frequency magnitude = sqrt(kx^2 + ky^2)  (proxy)
    speed_proxy = 1 / |k|   (only a proxy unless you map time frequency properly)

Note:
- True wave speed (m/s) needs careful treatment of phase wrapping, band center
  frequency, and sometimes surface geodesic distances. This app focuses on a
  robust *direction/consistency* pipeline with clear hooks for upgrades.
"""

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

import mne
from scipy.signal import welch, hilbert

st.set_page_config(page_title="EEG Explorer", layout="wide")

# ----------------------------
# Utility: montage parsing
# ----------------------------

def parse_custom_positions(text: str):
    """
    Parse CSV-like lines: name,x,y,z
    Returns dict name -> np.array([x,y,z]) and list of errors.
    """
    pos = {}
    errors = []
    if not text.strip():
        return pos, errors

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for i, ln in enumerate(lines, start=1):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 4:
            errors.append(f"Línea {i}: esperaba 4 columnas 'name,x,y,z' pero recibí {len(parts)} -> {ln}")
            continue
        name = parts[0]
        try:
            x, y, z = map(float, parts[1:])
            pos[name] = np.array([x, y, z], dtype=float)
        except Exception:
            errors.append(f"Línea {i}: no pude convertir x,y,z a float -> {ln}")
    return pos, errors


def build_montage_for_raw(raw: mne.io.BaseRaw, montage_mode: str, custom_text: str):
    """
    Returns (raw_with_montage, used_names, warnings_list)
    - montage_mode: 'standard_1020' or 'custom'
    - custom_text: CSV with positions (optional)
    """
    warnings = []

    ch_names = raw.ch_names
    # Only EEG channels matter for montage; keep others intact
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    if len(eeg_names) == 0:
        warnings.append("No encontré canales EEG en el archivo (según MNE).")
        return raw, [], warnings

    if montage_mode == "standard_1020":
        std = mne.channels.make_standard_montage("standard_1020")
        # Some EDF channel labels include prefixes/suffixes (e.g., 'EEG Fp1-REF').
        # We'll attempt to map to canonical 10-20 labels by extracting tokens.
        def canonicalize(name):
            # pull common 10-20 token (Fp1, F3, Cz, etc.) if present
            m = re.search(r"\b([A-Za-z]{1,3}\d{0,2}|Cz|Fz|Pz|Oz|Fpz|CPz|FCz|POz|AFz|Iz)\b", name.replace("-", " "))
            return m.group(1) if m else name

        mapping = {}
        std_names = set(std.ch_names)
        used = []
        for nm in eeg_names:
            can = canonicalize(nm)
            if can in std_names:
                mapping[nm] = can
                used.append(nm)

        if len(mapping) == 0:
            warnings.append("No pude mapear tus nombres de canal a etiquetas 10-20 estándar. "
                            "Prueba a usar 'Custom' y pegar posiciones manualmente.")
            return raw, [], warnings

        # Rename a *copy* of the raw to avoid side effects
        raw2 = raw.copy()
        raw2.rename_channels(mapping)

        # Apply montage (only channels matching montage names get positions)
        raw2.set_montage(std, match_case=False, on_missing="warn")
        warnings.append(f"Montaje estándar aplicado. Canales mapeados: {len(mapping)} / {len(eeg_names)}.")
        return raw2, list(mapping.values()), warnings

    # Custom montage
    custom_pos, errors = parse_custom_positions(custom_text)
    if errors:
        warnings.extend(errors)

    if len(custom_pos) == 0:
        warnings.append("Modo 'Custom' seleccionado pero no se proporcionaron posiciones. "
                        "Pega al menos algunas líneas name,x,y,z.")
        return raw, [], warnings

    # Create montage from custom positions, but only for channels present
    present = {}
    for nm in eeg_names:
        if nm in custom_pos:
            present[nm] = custom_pos[nm]

    if len(present) == 0:
        warnings.append("Ninguna etiqueta del archivo coincide con las etiquetas del bloque custom. "
                        "Revisa mayúsculas/minúsculas y nombres exactos.")
        return raw, [], warnings

    montage = mne.channels.make_dig_montage(ch_pos=present, coord_frame="head")
    raw2 = raw.copy()
    raw2.set_montage(montage, on_missing="ignore")
    warnings.append(f"Montaje custom aplicado. Canales con posición: {len(present)} / {len(eeg_names)}.")
    return raw2, list(present.keys()), warnings


def project_xy(raw: mne.io.BaseRaw, picks):
    """
    Get 2D coordinates (x,y) for selected channels from raw.info.
    Returns coords array shape (n_channels, 2) and channel names.
    """
    chs = [raw.info["chs"][i] for i in picks]
    names = [ch["ch_name"] for ch in chs]
    xyz = []
    for ch in chs:
        loc = ch["loc"][:3].copy()
        if np.allclose(loc, 0):
            xyz.append([np.nan, np.nan, np.nan])
        else:
            xyz.append(loc)
    xyz = np.array(xyz, dtype=float)
    # simple orthographic projection: x,y
    xy = xyz[:, :2]
    return xy, names


# ----------------------------
# Feature extraction
# ----------------------------

BANDS = {
    "Delta (1-4 Hz)": (1.0, 4.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Alpha (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Gamma (30-45 Hz)": (30.0, 45.0),
}

def bandpower_from_psd(freqs, psd, fmin, fmax):
    """Integrate PSD within band (simple trapezoid)."""
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.nan
    return np.trapz(psd[idx], freqs[idx])

def extract_features_epochs(raw, epoch_len=2.0, overlap=0.5, bands=BANDS):
    """
    Slice raw EEG into overlapping epochs and extract bandpower features per channel.
    Returns:
      X: DataFrame (n_epochs x features)
      meta: DataFrame with epoch start time
    """
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels found.")

    data = raw.get_data(picks=picks)  # shape (n_channels, n_samples)
    n_ch, n_samp = data.shape

    win = int(epoch_len * sfreq)
    step = int(win * (1.0 - overlap))
    if step <= 0:
        step = win

    feats = []
    starts = []

    for start in range(0, n_samp - win + 1, step):
        seg = data[:, start:start+win]
        # Welch per channel
        f, pxx = welch(seg, fs=sfreq, nperseg=min(win, 2*int(sfreq)))
        row = {}
        for bi, (bname, (fmin, fmax)) in enumerate(bands.items()):
            bp = np.array([bandpower_from_psd(f, pxx[ch], fmin, fmax) for ch in range(n_ch)])
            for ci, ch_name in enumerate([raw.ch_names[i] for i in picks]):
                row[f"{ch_name}__{bname}"] = bp[ci]
        feats.append(row)
        starts.append(start / sfreq)

    X = pd.DataFrame(feats)
    meta = pd.DataFrame({"t_start_sec": starts})
    return X, meta


# ----------------------------
# Traveling waves
# ----------------------------

def estimate_wave_vector(phases, xy):
    """
    Estimate wave vector k = (kx, ky) in phase ~ kx*x + ky*y + c
    phases: (n_channels,) in radians
    xy: (n_channels,2) coordinates

    Steps:
    - Drop channels with NaN coords
    - Unwrap phases relative to circular mean to stabilize
    - Least squares fit
    """
    mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if mask.sum() < 4:
        return np.nan, np.nan

    ph = phases[mask].copy()
    coords = xy[mask]

    # Stabilize phase by referencing to circular mean angle
    ref = np.angle(np.mean(np.exp(1j * ph)))
    ph = np.angle(np.exp(1j * (ph - ref)))  # wrap to [-pi, pi)

    # Unwrap in a deterministic order to reduce discontinuities
    order = np.argsort(coords[:, 0] + 10*coords[:, 1])
    ph_u = np.unwrap(ph[order])
    coords_u = coords[order]

    A = np.c_[coords_u[:, 0], coords_u[:, 1], np.ones(coords_u.shape[0])]
    coef, *_ = np.linalg.lstsq(A, ph_u, rcond=None)
    kx, ky = coef[0], coef[1]
    return kx, ky


def traveling_waves(raw, fmin, fmax, tmin, tmax, picks=None):
    """
    Compute wave direction over time for a window.
    Returns DataFrame with time, kx, ky, direction_rad, k_norm, speed_proxy.
    """
    sfreq = raw.info["sfreq"]
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    # crop & filter
    raw_seg = raw.copy().crop(tmin=tmin, tmax=tmax)
    raw_seg.filter(l_freq=fmin, h_freq=fmax, picks=picks, verbose=False)

    data = raw_seg.get_data(picks=picks)  # (n_ch, n_samples)
    xy, names = project_xy(raw_seg, picks)

    analytic = hilbert(data, axis=1)
    phases = np.angle(analytic)  # (n_ch, n_samples)
    times = raw_seg.times

    rows = []
    for ti in range(phases.shape[1]):
        kx, ky = estimate_wave_vector(phases[:, ti], xy)
        if np.isnan(kx) or np.isnan(ky):
            rows.append((times[ti], np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        direction = np.arctan2(ky, kx)
        k_norm = np.sqrt(kx*kx + ky*ky)
        speed_proxy = np.nan if k_norm == 0 else 1.0 / k_norm
        rows.append((times[ti], kx, ky, direction, k_norm, speed_proxy))

    return pd.DataFrame(rows, columns=["time_sec", "kx", "ky", "direction_rad", "k_norm", "speed_proxy"])


# ----------------------------
# UI
# ----------------------------

st.title("EEG Explorer (sklearn) + Traveling Waves (montage-aware)")

with st.sidebar:
    st.header("1) Datos")
    uploaded = st.file_uploader("Sube un EDF", type=["edf"])
    st.caption("Si tu EDF trae canales con nombres tipo 'EEG Fp1-REF', el modo estándar intentará mapearlos.")

    st.header("Montaje / posiciones")
    montage_mode = st.selectbox("Modo", ["standard_1020", "custom"], index=0)
    custom_text = ""
    if montage_mode == "custom":
        custom_text = st.text_area(
            "Custom electrode positions (CSV: name,x,y,z)",
            height=180,
            placeholder="Fp1,-0.0294,0.0836,0.0310\nFp2,0.0294,0.0836,0.0310\nFz,0.0,0.06,0.075\n..."
        )
        st.caption("Consejo: pega solo los canales que quieres usar en traveling waves.")

    st.header("2) Features")
    epoch_len = st.slider("Longitud de epoch (s)", 0.5, 10.0, 2.0, 0.5)
    overlap = st.slider("Solape", 0.0, 0.9, 0.5, 0.1)

    st.header("3) ML (5 modelos)")
    task = st.selectbox("Tipo de problema", ["Clasificación", "Regresión"], index=0)
    cv_folds = st.slider("CV folds", 3, 10, 5, 1)

    st.header("Traveling waves")
    band_name = st.selectbox("Banda", list(BANDS.keys()), index=2)
    tw_downsample = st.slider("Downsample (solo para plots)", 1, 20, 5, 1)

if not uploaded:
    st.info("Sube un EDF para empezar.")
    st.stop()

# Load raw
try:
    raw = mne.io.read_raw_edf(io.BytesIO(uploaded.getvalue()), preload=True, verbose=False)
except Exception as e:
    st.error(f"No pude leer el EDF: {e}")
    st.stop()

# Apply montage
raw_m, used_montage_names, mont_warnings = build_montage_for_raw(raw, montage_mode, custom_text)
for w in mont_warnings:
    st.warning(w)

# Basic info
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Canales", len(raw_m.ch_names))
with colB:
    st.metric("Duración (s)", f"{raw_m.times[-1]:.1f}")
with colC:
    st.metric("Fs (Hz)", f"{raw_m.info['sfreq']:.1f}")

# ----------------------------
# Visual: PSD
# ----------------------------
st.subheader("Análisis de frecuencia (PSD)")
picks_eeg = mne.pick_types(raw_m.info, eeg=True, exclude=[])
if len(picks_eeg) == 0:
    st.error("No hay canales EEG detectados.")
    st.stop()

# pick first N channels for PSD plot to keep it readable
n_show = min(6, len(picks_eeg))
show_picks = picks_eeg[:n_show]
data = raw_m.get_data(picks=show_picks)
sfreq = raw_m.info["sfreq"]

fig = plt.figure()
for i in range(data.shape[0]):
    f, pxx = welch(data[i], fs=sfreq, nperseg=min(len(data[i]), int(2*sfreq)))
    plt.semilogy(f, pxx, label=raw_m.ch_names[show_picks[i]])
plt.xlim(0, 50)
plt.xlabel("Hz")
plt.ylabel("PSD")
plt.legend()
st.pyplot(fig)

# ----------------------------
# Feature extraction
# ----------------------------
st.subheader("Características (bandpower por epoch) + PCA")
try:
    X, meta = extract_features_epochs(raw_m, epoch_len=epoch_len, overlap=overlap)
except Exception as e:
    st.error(f"Error extrayendo features: {e}")
    st.stop()

st.write(f"Epochs: {len(X)} | Features: {X.shape[1]}")
st.dataframe(X.head(10))

# PCA plot
pipe_pca = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2))])
Z = pipe_pca.fit_transform(X.values)
fig = plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], s=12)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA de features (sin target)")
st.pyplot(fig)

# ----------------------------
# ML: pick target & run CV among 5 models
# ----------------------------
st.subheader("Entrenamiento y selección de modelo (5 opciones)")

target_mode = st.radio("Origen del target", ["Columna existente en features (si aplica)", "Crear target manual (demo)"], index=1)
y = None

if target_mode == "Columna existente en features (si aplica)":
    target_col = st.selectbox("Elige columna target", list(X.columns))
    y = X[target_col].values
    X_ml = X.drop(columns=[target_col])
else:
    st.caption("Demo: genera un target a partir de una combinación de bandas. Útil para probar el pipeline.")
    # simple synthetic target from a band average
    alpha_cols = [c for c in X.columns if "Alpha" in c]
    base = np.nanmean(X[alpha_cols].values, axis=1) if alpha_cols else np.nanmean(X.values, axis=1)
    if task == "Clasificación":
        thr = np.nanmedian(base)
        y = (base > thr).astype(int)
    else:
        y = base.astype(float)
    X_ml = X.copy()

# Define 5 models
models = {}
if task == "Clasificación":
    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "SVM (RBF)": SVC(kernel="rbf"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "RidgeClassifier-ish": LogisticRegression(max_iter=2000, penalty="l2"),  # keeps it simple
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="weighted")
    metric_name = "F1 (weighted)"
else:
    models = {
        "Ridge": Ridge(),
        "SVR (RBF)": SVR(kernel="rbf"),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Linear (Ridge alt)": Ridge(alpha=1.0),
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = make_scorer(r2_score)
    metric_name = "R²"

# Evaluate
results = []
for name, model in models.items():
    est = Pipeline([("scaler", StandardScaler()), ("model", model)])
    try:
        scores = cross_val_score(est, X_ml.values, y, cv=cv, scoring=scorer, n_jobs=None)
        results.append((name, float(np.mean(scores)), float(np.std(scores))))
    except Exception as e:
        results.append((name, np.nan, np.nan))
        st.warning(f"Modelo {name} falló: {e}")

res_df = pd.DataFrame(results, columns=["Modelo", f"{metric_name}_mean", f"{metric_name}_std"]).sort_values(by=f"{metric_name}_mean", ascending=False)
st.dataframe(res_df, use_container_width=True)

best = res_df.iloc[0]["Modelo"]
st.success(f"Mejor modelo según {metric_name}: **{best}**")

# ----------------------------
# Traveling waves
# ----------------------------
st.subheader("Traveling waves (requiere posiciones de electrodos)")
fmin, fmax = BANDS[band_name]

duration = float(raw_m.times[-1])
tmin = st.slider("tmin (s)", 0.0, max(0.0, duration-1.0), 0.0, 0.5)
tmax = st.slider("tmax (s)", min(1.0, duration), duration, min(duration, max(tmin+5.0, 10.0)), 0.5)

# picks for traveling waves: only channels with known positions
picks_tw = mne.pick_types(raw_m.info, eeg=True, exclude=[])
xy, names = project_xy(raw_m, picks_tw)
mask_known = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
known_count = int(mask_known.sum())

st.write(f"Canales EEG con coordenadas válidas: **{known_count} / {len(picks_tw)}**")
if known_count < 6:
    st.warning("Muy pocos canales con coordenadas para estimar traveling waves. "
               "Usa 'standard_1020' con nombres 10-20 o pega posiciones custom.")
else:
    tw_df = traveling_waves(raw_m, fmin, fmax, tmin, tmax, picks=[p for p, ok in zip(picks_tw, mask_known) if ok])
    # Downsample for plotting
    tw_plot = tw_df.iloc[::tw_downsample].copy()

    # Direction plot
    fig = plt.figure()
    plt.plot(tw_plot["time_sec"], tw_plot["direction_rad"])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Dirección (rad)")
    plt.title(f"Dirección de traveling wave ({band_name})")
    st.pyplot(fig)

    # k_norm plot
    fig = plt.figure()
    plt.plot(tw_plot["time_sec"], tw_plot["k_norm"])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("|k| (rad / unidad-espacial)")
    plt.title("Magnitud del vector de onda (proxy)")
    st.pyplot(fig)

    st.dataframe(tw_df.head(20), use_container_width=True)

st.caption("Siguiente mejora típica: estimar vecindades por triangulación en el scalp y hacer ajuste circular robusto. "
           "Este pipeline ya te deja el montaje listo y la estructura para iterar.")
