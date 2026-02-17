# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from src.eeg_io import EEGData, load_edf_bytes_to_eegdata
from src.features import welch_psd, bandpower_from_psd, DEFAULT_BANDS
from src.features import window_bandpower_features
from src.features import pca_channels
from src.viz import plot_topomap_band
from src.montage import parse_custom_montage_csv, get_standard_coords
from src.waves import traveling_wave_metrics
from src.ml import generate_lr_labels, evaluate_models
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="EEG App", layout="wide")
st.title("EEG Analysis (Streamlit)")

# =========================
# 1) Upload con límite + warning
# =========================
MAX_MB = 50
HARD_MAX_MB = 200

uploaded = st.file_uploader("Sube un EDF/EDF+ (.edf)", type=["edf"])

if uploaded is None:
    st.info("Sube un EDF para empezar.")
    st.stop()

size_mb = uploaded.size / (1024 * 1024)

if uploaded.size > MAX_MB * 1024 * 1024:
    st.warning(
        f"Archivo grande: {size_mb:.1f} MB. "
        f"Recomendado <= {MAX_MB} MB para la demo online. "
        "Puede ir lento."
    )

if uploaded.size > HARD_MAX_MB * 1024 * 1024:
    st.error(f"Archivo demasiado grande: {size_mb:.1f} MB. Límite: {HARD_MAX_MB} MB.")
    st.stop()

# =========================
# 2) Selector de recorte (primeros X segundos)
# =========================
crop_on = st.checkbox("Recortar para la demo (primeros X segundos)", value=True)
crop_sec = st.slider("Segundos a conservar", 5, 300, 60, step=5, disabled=not crop_on)


# =========================
# 3) Caching: carga + recorte
# =========================
@st.cache_data(show_spinner=False)
def load_cached(edf_bytes: bytes, crop_on: bool, crop_sec: float) -> EEGData:
    return load_edf_bytes_to_eegdata(
        edf_bytes=edf_bytes,
        crop_on=crop_on,
        crop_sec=float(crop_sec),
    )


with st.spinner("Cargando EDF (con cache)…"):
    eeg = load_cached(uploaded.getvalue(), crop_on, float(crop_sec))

st.success(
    f"Cargado ✅ {len(eeg.ch_names)} canales | {eeg.times[-1]:.1f}s | "
    f"sfreq={eeg.sfreq:.2f} Hz | {eeg.X.shape[0]} muestras"
)

# =========================
# Sidebar: selección de canales
# =========================
st.sidebar.header("Selección de canales EEG")

default_1020 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

available = eeg.ch_names
default = [ch for ch in default_1020 if ch in available]

selected_channels = st.sidebar.multiselect(
    "Canales a usar", options=available, default=default if default else available
)

if len(selected_channels) < 2:
    st.error("Selecciona al menos 2 canales EEG")
    st.stop()

# Filtrar EEG a canales seleccionados
idx = [eeg.ch_names.index(ch) for ch in selected_channels]
eeg.X = eeg.X[:, idx]
eeg.ch_names = [eeg.ch_names[i] for i in idx]

# (Opcional pero recomendable) actualizar times sigue igual; sfreq no cambia

# =========================
# Pre-cálculos globales (CACHEADOS)
# =========================


@st.cache_data(show_spinner=False)
def cached_welch_psd(X: np.ndarray, sfreq: float, nperseg: int):
    freqs, psd = welch_psd(X, sfreq, nperseg=nperseg)
    return freqs, psd


@st.cache_data(show_spinner=False)
def cached_bandpower(freqs, psd, bands):
    return bandpower_from_psd(freqs, psd, bands)


# Ajusta nperseg según longitud real
nper = min(2048, eeg.X.shape[0])

freqs_all, psd_all = cached_welch_psd(eeg.X, eeg.sfreq, nper)
bp = cached_bandpower(freqs_all, psd_all, DEFAULT_BANDS)

df_bp = pd.DataFrame(bp, index=eeg.ch_names)


tab1, tab2, tab3, tab4 = st.tabs(
    ["Exploración", "Features/PCA", "ML", "Traveling Waves"]
)

with tab1:
    st.subheader("Señal (primeros segundos)")

    ch_plot = st.selectbox("Canal a visualizar", eeg.ch_names, index=0)
    ch_i = eeg.ch_names.index(ch_plot)

    seconds = st.slider("Segundos a mostrar", 1, 30, 10)
    n = min(int(seconds * eeg.sfreq), eeg.X.shape[0])

    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
    ax.plot(eeg.times[:n], eeg.X[:n, ch_i])
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title(f"Canal: {ch_plot}")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("PSD (Welch)")

    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.semilogy(freqs_all, psd_all[ch_i])
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD")
    ax.set_title(f"PSD canal: {ch_plot}")
    fig.tight_layout()
    st.pyplot(fig)

    decim = max(1, int(eeg.sfreq / 200))
    ax.plot(eeg.times[:n:decim], eeg.X[:n:decim, ch_i])


# st.write("Nombres de canales:", eeg.ch_names)

with tab2:
    st.subheader("Bandpower por canal")

    nper = min(2048, eeg.X.shape[0])
    freqs, psd = welch_psd(eeg.X, eeg.sfreq, nperseg=nper)

    bp = bandpower_from_psd(freqs, psd, DEFAULT_BANDS)
    df_bp = pd.DataFrame(bp, index=eeg.ch_names)  # filas=canales, columnas=bandas

    # -------------------------
    # Tabla bandpower (solo visualización)
    # -------------------------
    log_table = st.checkbox("Tabla en log10", value=True)
    eps = 1e-12
    df_table = np.log10(df_bp + eps) if log_table else df_bp

    # -------------------------
    # PCA (sobre canales)
    # -------------------------
    st.subheader("PCA (sobre canales)")

    max_comp = max(1, int(min(eeg.X.shape[0], eeg.X.shape[1])))
    st.caption(f"PCA: n_canales={eeg.X.shape[1]} → máximo componentes = {max_comp}")

    ncomp = st.slider(
        "N componentes", min_value=1, max_value=max_comp, value=min(3, max_comp)
    )

    comps, evr = pca_channels(eeg.X, n_components=ncomp)
    st.write("Explained variance ratio:", evr)

    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
    ax.plot(eeg.times, comps[:, 0])
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("PC1 (serie temporal)")
    fig.tight_layout()
    st.pyplot(fig)

    # -------------------------
    # Tabla + Topomap en 2 columnas
    # -------------------------
    colL, colR = st.columns([1.2, 1.0], gap="large")

    with colL:
        st.subheader("Tabla bandpower")
        st.dataframe(
            df_table.style.format("{:.3f}" if log_table else "{:.2e}").set_properties(
                **{"text-align": "center"}
            )
        )

    with colR:
        st.subheader("Topomap (scalp map)")

        band_topo = st.selectbox(
            "Banda para topomap",
            list(DEFAULT_BANDS.keys()),
            index=list(DEFAULT_BANDS.keys()).index("alpha"),
        )

        mode = st.radio(
            "Escala del topomap",
            ["lineal", "log10", "z-score (espacial)"],
            index=1,
            help="lineal/log10 muestran magnitud; z-score muestra patrón relativo (media=0, sd=1).",
        )

        if mode == "lineal":
            st.info(
                "**Lineal**: potencia absoluta. Útil para ver magnitudes reales y detectar canales anómalos."
            )
        elif mode == "log10":
            st.info(
                "**Log10**: comprime rangos grandes. Útil si pocos canales dominan el mapa."
            )
        else:
            st.info(
                "**Z-score (espacial)**: muestra patrón relativo (media=0, sd=1). Pierde escala absoluta; ideal para comparar topografías."
            )

        if mode.startswith("z"):
            cmap = "RdBu_r"
            st.caption("Modo z-score: colormap fijado a RdBu_r (divergente).")
        else:
            cmap = st.selectbox(
                "Colormap", ["viridis", "plasma", "inferno", "magma"], index=0
            )

        coords = get_standard_coords(eeg.ch_names)
        chs = [ch for ch in eeg.ch_names if ch in coords]

        if len(chs) < 4:
            st.warning(
                f"No hay suficientes canales con coordenadas para topomap: {len(chs)} / {len(eeg.ch_names)}"
            )
            missing = [ch for ch in eeg.ch_names if ch not in coords]
            if missing:
                st.caption("Sin coordenadas: " + ", ".join(missing))
        else:
            values = np.array(
                [float(df_bp.loc[ch, band_topo]) for ch in chs], dtype=float
            )

            if mode == "log10":
                values = np.log10(values + 1e-12)
            elif mode.startswith("z"):
                mu = float(np.mean(values))
                sd = float(np.std(values) + 1e-12)
                values = (values - mu) / sd

            vlim = (-3, 3) if mode.startswith("z") else None

            info = mne.create_info(
                ch_names=chs, sfreq=eeg.sfreq, ch_types=["eeg"] * len(chs)
            )
            mont = mne.channels.make_dig_montage(
                {ch: coords[ch] for ch in chs}, coord_frame="head"
            )
            info.set_montage(mont)

            fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=140)

            if vlim is None:
                ret = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap)
            else:
                try:
                    ret = mne.viz.plot_topomap(
                        values, info, axes=ax, show=False, cmap=cmap, vlim=vlim
                    )
                except TypeError:
                    ret = mne.viz.plot_topomap(
                        values,
                        info,
                        axes=ax,
                        show=False,
                        cmap=cmap,
                        vmin=vlim[0],
                        vmax=vlim[1],
                    )

            im = ret[0] if isinstance(ret, tuple) else ret
            fig.colorbar(im, ax=ax, shrink=0.75)

            pos = np.array([coords[ch][:2] for ch in chs], dtype=float)
            for (x, y), ch in zip(pos, chs):
                ax.text(x, y, ch, fontsize=7, ha="center", va="center")

            if mode.startswith("z"):
                ax.set_title(f"{band_topo} -z - score espacial")
            else:
                ax.set_title(f"{band_topo} - {mode}")

            fig.tight_layout()
            st.pyplot(fig)

            # botón de descarga PNG
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button(
                "Descargar topomap (PNG)",
                data=buf.getvalue(),
                file_name=f"topomap_{band_topo}_{mode}.png",
                mime="image/png",
            )

    # =========================================================
    # Comparación A vs B (topomaps) ✅ (AQUÍ ya estamos a nivel tab2)
    # =========================================================
    st.subheader("Comparación A vs B (topomaps)")

    from scipy.stats import ttest_ind

    def parse_intervals_ab(text: str):
        intervals = []
        text = (text or "").replace(";", "\n").replace("/", "\n")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                t0, t1 = float(parts[0]), float(parts[1])
                lab = parts[2].upper()
            except Exception:
                continue
            if lab not in ("A", "B"):
                continue
            if t1 <= t0:
                continue
            intervals.append((t0, t1, lab))
        return intervals

    def cohen_d_independent(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        muA = np.nanmean(A, axis=0)
        muB = np.nanmean(B, axis=0)
        sA = np.nanstd(A, axis=0, ddof=1)
        sB = np.nanstd(B, axis=0, ddof=1)
        nA = A.shape[0]
        nB = B.shape[0]
        sp = np.sqrt(((nA - 1) * (sA**2) + (nB - 1) * (sB**2)) / max(1, (nA + nB - 2)))
        return (muA - muB) / (sp + 1e-12)

    def fdr_bh(pvals: np.ndarray, alpha: float = 0.05):
        p = np.asarray(pvals, float)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        thresh = alpha * (np.arange(1, m + 1) / m)

        reject = ranked <= thresh
        if not np.any(reject):
            return np.zeros_like(p, dtype=bool), np.full_like(p, np.nan)

        kmax = np.max(np.where(reject)[0])
        cutoff = ranked[kmax]
        reject_mask = p <= cutoff

        p_adj = np.empty_like(ranked)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            val = ranked[i] * m / (i + 1)
            prev = min(prev, val)
            p_adj[i] = prev

        p_adj_full = np.empty_like(p)
        p_adj_full[order] = p_adj
        return reject_mask, p_adj_full

    st.caption(
        "Formato: t0,t1,label (seg). label debe ser A o B. Separadores: líneas, ';' o '/'. Ej: 0,30,A / 30,60,B"
    )

    with st.expander("⚙ Intervalos automáticos (recomendado para demo)", expanded=True):
        auto_on = st.checkbox(
            "Generar intervalos automáticamente", value=False, key="ab_auto_on"
        )
        auto_mode = st.selectbox(
            "Modo",
            [
                "Mitad A / Mitad B",
                "Alternancia por bloques (A,B,A,B...)",
                "N bloques iguales + asignación manual",
            ],
            index=0,
            disabled=not auto_on,
            key="ab_auto_mode",
        )

        t_total = float(eeg.times[-1])
        if auto_on:
            if auto_mode == "Mitad A / Mitad B":
                mid = t_total / 2
                auto_intervals = [(0.0, mid, "A"), (mid, t_total, "B")]
            elif auto_mode == "Alternancia por bloques (A,B,A,B...)":
                block_sec = st.number_input(
                    "Duración de bloque (s)",
                    0.5,
                    60.0,
                    5.0,
                    0.5,
                    key="ab_block_sec",
                    disabled=not auto_on,
                )
                auto_intervals = []
                t0i = 0.0
                k = 0
                while t0i < t_total:
                    t1i = min(t_total, t0i + float(block_sec))
                    lab = "A" if (k % 2 == 0) else "B"
                    auto_intervals.append((t0i, t1i, lab))
                    t0i = t1i
                    k += 1
            else:
                n_blocks = st.slider(
                    "N bloques", 2, 20, 6, key="ab_n_blocks", disabled=not auto_on
                )
                block_len = t_total / int(n_blocks)
                pattern = st.text_input(
                    "Patrón A/B (ej: AABBAB)",
                    value="AB" * (int(n_blocks) // 2),
                    key="ab_pattern",
                    disabled=not auto_on,
                )
                pattern = (pattern.upper().replace(" ", ""))[: int(n_blocks)]
                if len(pattern) < int(n_blocks):
                    pattern = (pattern + ("AB" * 50))[: int(n_blocks)]

                auto_intervals = []
                for i in range(int(n_blocks)):
                    t0i = i * block_len
                    t1i = (i + 1) * block_len if i < int(n_blocks) - 1 else t_total
                    lab = "A" if pattern[i] == "A" else "B"
                    auto_intervals.append((t0i, t1i, lab))

            st.write("Intervalos generados:")
            st.code("\n".join([f"{a:.2f},{b:.2f},{c}" for a, b, c in auto_intervals]))
            ab_text_default = "\n".join(
                [f"{a:.4f},{b:.4f},{c}" for a, b, c in auto_intervals]
            )
        else:
            ab_text_default = ""

    ab_text = st.text_area(
        "Intervalos A/B", value=ab_text_default, height=140, key="ab_text"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        win_sec_ab = st.number_input(
            "Ventana A/B (s)", 0.5, 30.0, 2.0, 0.5, key="win_ab"
        )
    with col2:
        step_sec_ab = st.number_input(
            "Paso A/B (s)", 0.1, 30.0, 1.0, 0.1, key="step_ab"
        )
    with col3:
        band_ab = st.selectbox(
            "Banda",
            list(DEFAULT_BANDS.keys()),
            index=list(DEFAULT_BANDS.keys()).index("alpha"),
            key="band_ab",
        )
        lo_ab, hi_ab = DEFAULT_BANDS[band_ab]

    intervals = parse_intervals_ab(ab_text)
    ab_stats_ready = False
    if not intervals:
        st.info("Pega intervalos A/B o activa intervalos automáticos.")
    else:

        Xw, t_centers_ab, _ = window_bandpower_features(
            eeg.X,
            eeg.sfreq,
            eeg.ch_names,
            bands={band_ab: (lo_ab, hi_ab)},
            win_sec=float(win_sec_ab),
            step_sec=float(step_sec_ab),
            nperseg=512,
        )

        coords_ab = get_standard_coords(eeg.ch_names)
        chs_ab = [ch for ch in eeg.ch_names if ch in coords_ab]
        if len(chs_ab) < 4:
            st.warning(
                f"No hay suficientes canales con coordenadas para topomap: {len(chs_ab)} / {len(eeg.ch_names)}"
            )
        else:

            idx_ch = [eeg.ch_names.index(ch) for ch in chs_ab]
            Xw_ch = Xw[:, idx_ch]

            y_ab = np.array(["" for _ in t_centers_ab], dtype=object)
            for t0i, t1i, lab in intervals:
                mask = (t_centers_ab >= t0i) & (t_centers_ab < t1i)
                y_ab[mask] = lab

            keep = y_ab != ""
            Xw_ch = Xw_ch[keep]
            y_ab = y_ab[keep]
            t_keep = t_centers_ab[keep]

            with st.expander("🧪 Debug: etiquetas por ventana", expanded=False):
                fig = plt.figure(figsize=(7, 2), dpi=130)
                classes = {"A": 0, "B": 1}
                yy = np.array([classes[v] for v in y_ab], int)
                plt.plot(t_keep, yy, marker=".", linestyle="none")
                plt.yticks([0, 1], ["A", "B"])
                plt.xlabel("Tiempo (s)")
                plt.title("Etiquetas por ventana (A/B)")
                st.pyplot(fig)

            A = Xw_ch[y_ab == "A"]
            B = Xw_ch[y_ab == "B"]
            if len(A) < 2 or len(B) < 2:
                st.warning(
                    f"Necesitas >=2 ventanas por clase. Ahora: A={len(A)}, B={len(B)}."
                )
            else:
                muA = np.mean(A, axis=0)
                muB = np.mean(B, axis=0)
                diff = muA - muB
                d = cohen_d_independent(A, B)
                t_stat, pvals = ttest_ind(
                    A, B, axis=0, equal_var=False, nan_policy="omit"
                )
                ab_stats_ready = True

    if ab_stats_ready:
        colS1, colS2 = st.columns([1.2, 1.0])
        with colS1:
            alpha = st.slider(
                "α (significación)", 0.001, 0.2, 0.05, step=0.005, key="ab_alpha"
            )
        with colS2:
            corr = st.selectbox(
                "Corrección",
                ["sin corregir", "FDR (Benjamini–Hochberg)"],
                index=1,
                key="ab_corr",
            )

        if corr.startswith("FDR"):
            sig_mask, p_adj = fdr_bh(pvals, alpha=float(alpha))
            p_show = p_adj
        else:
            sig_mask = pvals < float(alpha)
            p_show = pvals

        st.subheader("Mapas A / B / A−B / Cohen’s d / p-values")

        mode_ab = st.radio(
            "Escala A y B", ["lineal", "log10"], index=1, horizontal=True, key="mode_ab"
        )

        A_map = muA.copy()
        B_map = muB.copy()
        D_map = diff.copy()

        if mode_ab == "log10":
            A_map = np.log10(A_map + 1e-12)
            B_map = np.log10(B_map + 1e-12)
            D_map = A_map - B_map

        p_plot = -np.log10(p_show + 1e-300)

        c1, c2, c3 = st.columns(3)
        with c1:
            figA = plot_topomap_band(
                A_map,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title=f"A ({band_ab})",
                cmap="viridis",
                vlim=None,
                figsize=(2.8, 2.8),
            )
            st.pyplot(figA)
        with c2:
            figB = plot_topomap_band(
                B_map,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title=f"B ({band_ab})",
                cmap="viridis",
                vlim=None,
                figsize=(2.8, 2.8),
            )
            st.pyplot(figB)
        with c3:
            vmax = float(np.nanmax(np.abs(D_map))) + 1e-12
            figD = plot_topomap_band(
                D_map,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title="A − B",
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                figsize=(2.8, 2.8),
            )
            st.pyplot(figD)

        c4, c5, c6 = st.columns(3)
        with c4:
            vmax_d = float(np.nanmax(np.abs(d))) + 1e-12
            figd = plot_topomap_band(
                d,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title="Cohen’s d",
                cmap="RdBu_r",
                vlim=(-vmax_d, vmax_d),
                figsize=(2.8, 2.8),
            )
            st.pyplot(figd)

        with c5:
            vmax_p = float(np.nanmax(p_plot)) + 1e-12
            figp = plot_topomap_band(
                p_plot,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title="-log10(p)",
                cmap="magma",
                vlim=(0.0, vmax_p),
                figsize=(2.8, 2.8),
            )
            st.pyplot(figp)

        with c6:
            d_sig = d.copy()
            d_sig[~sig_mask] = 0.0
            vmax_ds = float(np.nanmax(np.abs(d_sig))) + 1e-12
            figds = plot_topomap_band(
                d_sig,
                chs_ab,
                coords_ab,
                eeg.sfreq,
                title="d (solo significativo)",
                cmap="RdBu_r",
                vlim=(-vmax_ds, vmax_ds),
                figsize=(2.8, 2.8),
            )
            st.pyplot(figds)

        st.caption(
            "Interpretación:\n"
            "- **A−B**: diferencia cruda (o log-ratio si estás en log10).\n"
            "- **Cohen’s d**: tamaño de efecto normalizado.\n"
            "- **-log10(p)**: significación (más alto = p más pequeño).\n"
            "- **d solo significativo**: efecto mostrado solo donde pasa el umbral."
        )

with tab3:
    st.subheader("Entrenamiento ML (por ventanas temporales)")
    st.caption(
        "Cada fila = una ventana temporal. Features = bandpower por canal y banda."
    )

    # --- Parámetros de ventana ---
    colA, colB, colC = st.columns(3)
    with colA:
        win_sec = st.number_input(
            "Ventana (s)",
            min_value=0.5,
            max_value=30.0,
            value=2.0,
            step=0.5,
            key="ml_win",
        )
    with colB:
        step_sec = st.number_input(
            "Paso/solape (s)",
            min_value=0.1,
            max_value=30.0,
            value=1.0,
            step=0.1,
            key="ml_step",
        )
    with colC:
        nperseg = st.number_input(
            "Welch nperseg",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            key="ml_nper",
        )

    if step_sec > win_sec:
        st.warning(
            "Paso mayor que ventana: tendrás pocas ventanas. Normalmente step <= win."
        )

    X_feat, t_centers, feat_names = window_bandpower_features(
        eeg.X,
        eeg.sfreq,
        eeg.ch_names,
        bands=DEFAULT_BANDS,
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        nperseg=int(nperseg),
    )

    with st.expander("Ver nombres de features (feat_names)", expanded=False):
        st.dataframe(
            pd.DataFrame({"feature": feat_names}).head(50), use_container_width=True
        )

    st.write("Nº features:", len(feat_names))
    # st.write("feat_names[:10]:", feat_names[:10])

    if X_feat.shape[0] < 2:
        st.error("No hay suficientes ventanas. Reduce win_sec o step_sec.")
        st.stop()

    st.write("X_feat (ventanas x features):", X_feat.shape)
    st.write("Nº ventanas:", X_feat.shape[0])

    # --- TW features (opcional) ---
    use_tw = st.checkbox(
        "Añadir traveling-wave metrics como features", value=False, key="ml_use_tw"
    )
    if use_tw:
        tw_band = st.selectbox(
            "Banda para traveling waves",
            list(DEFAULT_BANDS.keys()),
            index=list(DEFAULT_BANDS.keys()).index("alpha"),
            key="ml_tw_band",
        )
        lo_tw, hi_tw = DEFAULT_BANDS[tw_band]

        coords = get_standard_coords(eeg.ch_names)
        if len(coords) < 4:
            st.error("No hay suficientes canales con coordenadas para TW features.")
            st.stop()

        tmax = float(eeg.times[-1])

        key = ("tw_full", tuple(eeg.ch_names), float(eeg.sfreq), tw_band)
        if key not in st.session_state:
            st.session_state[key] = traveling_wave_metrics(
                X=eeg.X,
                sfreq=eeg.sfreq,
                ch_names=eeg.ch_names,
                coords=coords,
                band=(lo_tw, hi_tw),
                t0=0.0,
                t1=tmax,
            )
        out_full = st.session_state[key]

        times_tw = np.asarray(out_full["times"], float)
        kmag_tw = np.asarray(out_full["k_mag"], float)
        spd_tw = np.asarray(out_full["speed_proxy"], float)

        kx_tw = np.asarray(out_full.get("kx", np.full_like(kmag_tw, np.nan)), float)
        ky_tw = np.asarray(out_full.get("ky", np.full_like(kmag_tw, np.nan)), float)

        win = float(win_sec)

        tw_feats = np.zeros((len(t_centers), 5), dtype=float)
        for i, tc in enumerate(t_centers):
            a = tc - win / 2
            b = tc + win / 2
            i0 = np.searchsorted(times_tw, a, side="left")
            i1 = np.searchsorted(times_tw, b, side="right")
            if i1 <= i0:
                continue
            tw_feats[i, 0] = float(np.nanmean(kmag_tw[i0:i1]))
            tw_feats[i, 1] = float(np.nanmean(spd_tw[i0:i1]))
            tw_feats[i, 2] = float(np.nanmean(kx_tw[i0:i1]))
            tw_feats[i, 3] = float(np.nanmean(ky_tw[i0:i1]))
            tw_feats[i, 4] = float(np.nanstd(kmag_tw[i0:i1]))

        tw_names = [
            "tw_kmag_mean",
            "tw_speed_mean",
            "tw_kx_mean",
            "tw_ky_mean",
            "tw_kmag_std",
        ]
        X_feat = np.hstack([X_feat, tw_feats])
        feat_names = feat_names + tw_names
        st.write("X_feat + TW:", X_feat.shape)

    # --- Spatial features (opcional) ---
    use_spatial = st.checkbox(
        "Añadir spatial features (AP/LR/GFP/CoM)", value=False, key="ml_use_spatial"
    )
    if use_spatial:
        from src.spatial import spatial_features_from_X_feat

        band_names = list(DEFAULT_BANDS.keys())
        target_band = st.selectbox(
            "Banda para spatial features",
            band_names,
            index=band_names.index("alpha") if "alpha" in band_names else 0,
            key="ml_spatial_band",
        )

        coords_sp = get_standard_coords(eeg.ch_names)
        S, s_names = spatial_features_from_X_feat(
            X_feat=X_feat,
            ch_names=eeg.ch_names,
            band_names=band_names,
            target_band=target_band,
            coords=coords_sp,
        )

        if S.shape[1] == 0:
            st.warning(
                "No se pudieron calcular spatial features (faltan canales típicos o coords)."
            )
        else:
            X_feat = np.hstack([X_feat, S])
            feat_names = feat_names + s_names
            st.write("X_feat + spatial:", X_feat.shape)
            st.caption(
                f"Spatial features calculadas sobre {target_band} (mapa canal×banda por ventana)."
            )

    # --- Tipo de tarea ---
    task = st.selectbox(
        "Tipo de tarea", ["classification", "regression"], key="ml_task"
    )

    # --- Etiquetas (y) por ventana ---
    st.subheader("Etiquetas (y) por ventana")
    st.caption(f"Rango ventanas: {t_centers[0]:.2f}s – {t_centers[-1]:.2f}s")

    label_mode = st.selectbox(
        "Cómo definir y",
        [
            "demo (primera mitad=0, segunda mitad=1)",
            "manual (pegar lista)",
            "manual por intervalos (t0,t1,label)",
        ],
        key="ml_label_mode",
    )

    y = None

    if label_mode.startswith("demo"):
        mid_t = float(t_centers[-1]) / 2.0
        y = np.array(["0" if t < mid_t else "1" for t in t_centers], dtype=object)
        st.caption(
            "Demo: útil solo para comprobar que el pipeline funciona (no es ciencia)."
        )

    elif label_mode.startswith("manual (pegar lista)"):
        st.caption(
            "Pega una etiqueta por ventana (mismo número que ventanas), separadas por comas."
        )
        y_text = st.text_area("y", value="", height=120, key="ml_y_text")
        if y_text.strip():
            y = np.array(
                [s.strip() for s in y_text.split(",") if s.strip() != ""], dtype=object
            )

    else:
        st.caption(
            "Pega intervalos: una línea por intervalo -> t0,t1,label (segundos). Ej: 0,30,rest ; 30,60,task"
        )
        intervals = st.text_area("intervalos", value="", height=140, key="ml_intervals")
        if intervals.strip():
            y = np.array(["" for _ in t_centers], dtype=object)
            for line in intervals.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                t0i, t1i, lab = float(parts[0]), float(parts[1]), parts[2]
                mask = (t_centers >= t0i) & (t_centers < t1i)
                y[mask] = lab

            keep = y != ""
            X_feat = X_feat[keep]
            t_centers = t_centers[keep]
            y = y[keep]
            st.write("Ventanas etiquetadas:", int(len(y)))

            if X_feat.shape[0] < 2:
                st.error("No hay suficientes ventanas etiquetadas para entrenar.")
                st.stop()

    # ✅ DEBUG etiquetas (opcional)
    if y is not None and len(y) == len(t_centers) and task == "classification":
        fig, ax = plt.subplots(figsize=(5, 1.8), dpi=120)

        classes = {c: i for i, c in enumerate(sorted(set(y.tolist())))}
        y_num = [classes[v] for v in y.tolist()]

        ax.plot(t_centers, y_num, marker=".", linestyle="none")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Clase (index)")
        ax.set_title("Etiquetas por ventana (debug)")

        ax.tick_params(labelsize=8)
        fig.tight_layout(pad=0.3)

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------
        # Controles Permutation Importance (FUERA del botón)
        # -------------------------
        # -------------------------
        # Importancia de variables (Permutation Importance)
        # -------------------------
        st.markdown("---")
        st.subheader("Qué variables aportan más información (Permutation Importance)")

        st.caption(
            "Importancia predictiva ≠ relevancia neurofisiológica."
            "Además, esta app NO hace limpieza avanzada de artefactos (ICA/ASR/etc.), "
            "así que algunas variables pueden capturar ruido (p. ej., EMG en beta)."
            "La importancia se calcula usando el modelo ganador con validación cruzada independiente."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            do_perm = st.checkbox("Calcular importancias", value=True, key="ml_do_perm")

        with col2:
            n_repeats = st.slider("Permutaciones", 5, 50, 20, 5, key="ml_perm_repeats")
        with col3:
            top_n = st.slider("Top N", 5, 50, 15, 1, key="ml_perm_topn")
        with col4:
            if task == "classification":
                scoring = st.selectbox(
                    "Scoring",
                    ["roc_auc", "accuracy", "f1_macro"],
                    index=0,
                    key="ml_perm_scoring",
                )
            else:
                scoring = st.selectbox(
                    "Scoring",
                    ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
                    index=0,
                    key="ml_perm_scoring",
                )

        st.caption(
            f"Scoring seleccionado: {st.session_state.get('ml_perm_scoring', 'roc_auc')}"
        )
        st.caption(f"Scoring seleccionado: {scoring}")

        # --- Entrenamiento ---
        if st.button("Evaluar modelos", key="ml_eval"):
            if y is None:
                st.error("Define y antes de entrenar.")
                st.stop()

            if len(y) != X_feat.shape[0]:
                st.error(
                    f"y ({len(y)}) debe tener la misma longitud que X_feat ({X_feat.shape[0]})."
                )
                st.stop()

            if task == "regression":
                y = y.astype(float)
                cv = min(5, max(2, X_feat.shape[0] // 10))
            else:
                if len(set(y.tolist())) < 2:
                    st.error(
                        f"Necesitas al menos 2 clases. Solo hay: {set(y.tolist())}"
                    )
                    st.stop()

                unique, counts = np.unique(y, return_counts=True)
                min_class = int(counts.min())
                cv = min(5, min_class)
                if cv < 2:
                    st.error(
                        f"No hay suficientes muestras por clase para CV. Recuentos: {dict(zip(unique, counts))}"
                    )
                    st.stop()

            st.caption(f"CV usado: {cv}")

            # 1) Evaluar modelos
            res = evaluate_models(X_feat, y, task=task, cv=cv)
            st.write("Scores:", res.scores)
            st.success(f"Mejor modelo: {res.best_name}")

            if do_perm:
                # 2) Recuperar el mejor estimador
                model = None
                if hasattr(res, "best_estimator") and res.best_estimator is not None:
                    model = res.best_estimator
                elif hasattr(res, "best_model") and res.best_model is not None:
                    model = res.best_model
                elif (
                    hasattr(res, "models")
                    and isinstance(res.models, dict)
                    and res.best_name in res.models
                ):
                    model = res.models[res.best_name]

                if model is None:
                    st.error(
                        "No puedo recuperar el modelo ganador desde `res`. "
                        "Solución rápida: haz que `evaluate_models` devuelva también `best_estimator`."
                    )
                    st.stop()

                # 3) Preparar X/y + nombres
                X = X_feat
                feature_names = list(feat_names)

                # 4) CV coherente
                if task == "classification":
                    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                else:
                    cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)

                # 5) Permutation importance por fold
                importances = []
                n_failed = 0

                for tr, te in cv_obj.split(X, y):
                    X_train, X_test = X[tr], X[te]
                    y_train, y_test = y[tr], y[te]

                    try:
                        model.fit(X_train, y_train)

                        pi = permutation_importance(
                            model,
                            X_test,
                            y_test,
                            n_repeats=int(n_repeats),
                            random_state=42,
                            scoring=scoring,
                        )
                        importances.append(pi.importances_mean)

                    except Exception as e:
                        n_failed += 1
                        st.warning(
                            f"Permutation importance falló (scoring='{scoring}') en un fold: {e}"
                        )

                if len(importances) == 0:
                    st.error(
                        f"No se pudo calcular permutation importance en ningún fold "
                        f"(fallaron {n_failed}/{cv}). Prueba otro scoring."
                    )
                    st.stop()

                if n_failed > 0:
                    st.info(
                        f"Permutation importance: {n_failed}/{cv} folds fallaron; "
                        "se promedia con los restantes."
                    )

                importances = np.asarray(importances)
                mean_imp = np.nanmean(importances, axis=0)
                std_imp = np.nanstd(importances, axis=0)

                df_imp = (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "importance_mean": mean_imp,
                            "importance_std": std_imp,
                        }
                    )
                    .sort_values("importance_mean", ascending=False)
                    .reset_index(drop=True)
                )

                # Guardar para descarga / evitar NameError en reruns
                st.session_state["ml_df_imp"] = df_imp

                # --- 1) Top variables (detalle fino) ---
                st.write("Top variables (media ± desviación entre folds):")

                # Si tu Streamlit lo soporta:
                st.dataframe(
                    df_imp.head(top_n), use_container_width=True, hide_index=True
                )

                # Si NO lo soporta, usa esta alternativa:
                # st.dataframe(df_imp.head(top_n).style.hide(axis="index"), use_container_width=True)

                st.bar_chart(df_imp.head(top_n).set_index("feature")["importance_mean"])

                # --- 2) Neuro-friendly: agregados por BANDA y por CANAL ---
                st.markdown(
                    "### Vista neuro-friendly: importancia por banda y por canal"
                )

                band_keys = list(DEFAULT_BANDS.keys())
                ch_set = set(eeg.ch_names)

                def parse_feature(feat: str):
                    s = feat.replace("-", "_").replace(" ", "_")
                    parts = [p for p in s.split("_") if p]
                    ch = next((p for p in parts if p in ch_set), None)
                    band = next((p for p in parts if p in band_keys), None)
                    return ch, band

                parsed = [parse_feature(f) for f in df_imp["feature"].tolist()]
                df_imp["ch"] = [p[0] for p in parsed]
                df_imp["band"] = [p[1] for p in parsed]

                df_cb = df_imp.dropna(subset=["ch", "band"]).copy()

                if len(df_cb) == 0:
                    st.info(
                        "No he podido inferir canal y banda desde los nombres de features, "
                        "así que no puedo agregar por banda/canal."
                    )
                else:
                    df_band = (
                        df_cb.groupby("band", as_index=False)
                        .agg(
                            importance_mean=("importance_mean", "sum"),
                            importance_std=("importance_std", "mean"),
                            n_features=("feature", "count"),
                        )
                        .sort_values("importance_mean", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.write(
                        "Importancia agregada por banda (suma de importancias medias):"
                    )
                    st.dataframe(df_band, use_container_width=True, hide_index=True)
                    st.bar_chart(df_band.set_index("band")["importance_mean"])

                    df_ch = (
                        df_cb.groupby("ch", as_index=False)
                        .agg(
                            importance_mean=("importance_mean", "sum"),
                            importance_std=("importance_std", "mean"),
                            n_features=("feature", "count"),
                        )
                        .sort_values("importance_mean", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.write(
                        "Importancia agregada por canal (suma de importancias medias):"
                    )
                    st.dataframe(
                        df_ch.head(20), use_container_width=True, hide_index=True
                    )
                    st.bar_chart(df_ch.head(20).set_index("ch")["importance_mean"])

                    # --- 3) Descarga (solo si existe df_imp) ---
                    df_imp_dl = st.session_state.get("ml_df_imp", None)

                    if df_imp_dl is not None:
                        st.download_button(
                            "Descargar importancias (CSV)",
                            data=df_imp_dl.to_csv(index=False).encode("utf-8"),
                            file_name=f"feature_importance_permutation_{task}_{st.session_state.get('ml_perm_scoring','scoring')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.info("Aún no hay importancias calculadas para descargar.")

with tab4:
    st.subheader("Traveling Waves (con montaje)")

    import time
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    min_needed = 4

    # -------------------------
    # Controles (montaje / banda / intervalo)
    # -------------------------
    montage_mode = st.radio(
        "Montaje",
        ["standard (MNE: 1005/1020)", "custom"],
        horizontal=True,
        key="tw_montage_mode",
    )

    if montage_mode.startswith("standard"):
        coords = get_standard_coords(eeg.ch_names)
        st.caption(f"Canales con coords estándar: {len(coords)} / {len(eeg.ch_names)}")
        missing = [ch for ch in eeg.ch_names if ch not in coords]
        if missing:
            st.caption("Sin coordenadas: " + ", ".join(missing))
    else:
        sample = (
            "# name,x,y,z\n"
            "Fp1,-0.0294,0.0836,0.0310\n"
            "Fp2,0.0294,0.0836,0.0310\n"
            "Fz,0.0,0.06,0.075\n"
            "Cz,0.0,0.0,0.09"
        )
        text = st.text_area(
            "Pega tu CSV de canales/coords",
            value=sample,
            height=140,
            key="tw_custom_csv",
        )
        coords = parse_custom_montage_csv(text)
        st.caption(f"Canales custom con coords: {len(coords)}")

    band_name = st.selectbox("Banda", list(DEFAULT_BANDS.keys()), key="tw_band")
    lo, hi = DEFAULT_BANDS[band_name]

    tmax = float(eeg.times[-1])
    cT0, cT1 = st.columns(2)
    with cT0:
        t0 = st.number_input("t0 (s)", value=0.0, step=1.0, key="tw_t0")
    with cT1:
        t1 = st.number_input("t1 (s)", value=min(10.0, tmax), step=1.0, key="tw_t1")

    # -------------------------
    # Reset / Calcular
    # -------------------------
    cR, cC = st.columns([1, 1])
    with cR:
        if st.button("Reset TW", key="tw_reset"):
            for k in ["tw_out", "tw_coords_used", "tw_band_used", "tw_phi_cache"]:
                st.session_state.pop(k, None)
            st.info(
                "Resultados TW borrados. Pulsa 'Calcular traveling waves' para recalcular."
            )
    with cC:
        calc = st.button("Calcular traveling waves", key="tw_calc")

    if calc:
        if len(coords) < min_needed:
            st.error(
                f"Necesitas al menos {min_needed} canales con coordenadas. Ahora: {len(coords)}."
            )
            st.stop()

        t0f = max(0.0, float(t0))
        t1f = min(tmax, float(t1))
        if t1f <= t0f:
            st.error("t1 debe ser mayor que t0.")
            st.stop()

        try:
            out = traveling_wave_metrics(
                X=eeg.X,
                sfreq=eeg.sfreq,
                ch_names=eeg.ch_names,
                coords=coords,
                band=(lo, hi),
                t0=t0f,
                t1=t1f,
            )
            st.session_state["tw_out"] = out
            st.session_state["tw_coords_used"] = coords
            st.session_state["tw_band_used"] = (band_name, lo, hi)
            st.session_state.pop("tw_phi_cache", None)  # invalida cache de fase
            st.success("TW calculado ✅. Usa el slider o activa Play.")
        except Exception as e:
            st.error(str(e))

    # -------------------------
    # Si no hay resultado, parar aquí
    # -------------------------
    if "tw_out" not in st.session_state:
        st.info(
            "Pulsa **Calcular traveling waves** para generar resultados y luego usa el slider/Play."
        )
        st.stop()

    out = st.session_state["tw_out"]
    coords_used = st.session_state.get("tw_coords_used", coords)
    band_name_used, lo_used, hi_used = st.session_state.get(
        "tw_band_used", (band_name, lo, hi)
    )

    used = list(out["used_channels"])
    xy = np.array([coords_used[ch][:2] for ch in used], dtype=float)
    cx, cy = xy.mean(axis=0)

    # -------------------------
    # Controles de animación / slider (FIX Streamlit)
    # -------------------------
    if "tw_ti_val" not in st.session_state:
        st.session_state["tw_ti_val"] = 0

    nT = len(out["times"])

    cA, cB, cD = st.columns([1.2, 1.0, 1.0])
    with cA:
        play = st.toggle("▶️ Play", value=False, key="tw_play")
    with cB:
        fps = st.slider("FPS", 1, 15, 6, key="tw_fps")
    with cD:
        step_i = st.slider("Paso (frames)", 1, 10, 1, key="tw_step_i")

    ti_slider = st.slider(
        "Instante (índice)",
        0,
        nT - 1,
        value=int(st.session_state["tw_ti_val"]),
        key="tw_ti_slider",
    )
    st.session_state["tw_ti_val"] = int(ti_slider)

    if play:
        next_ti = st.session_state["tw_ti_val"] + int(step_i)
        if next_ti >= nT:
            next_ti = 0
        st.session_state["tw_ti_val"] = int(next_ti)
        time.sleep(1.0 / max(1, int(fps)))
        st.rerun()

    ti = int(st.session_state["tw_ti_val"])
    t_vis = float(out["times"][ti])

    # -------------------------
    # Suavizado circular de theta (sin/cos)
    # -------------------------
    smooth_on = st.checkbox("Suavizar dirección (θ)", value=True, key="tw_smooth_on")
    smooth_win = st.slider(
        "Ventana suavizado (frames)", 1, 61, 11, step=2, key="tw_smooth_win"
    )

    theta_raw = np.asarray(out["direction_rad"], float)

    if smooth_on and smooth_win > 1:
        k = int(smooth_win)
        kernel = np.ones(k, dtype=float) / k
        c = np.convolve(np.cos(theta_raw), kernel, mode="same")
        s = np.convolve(np.sin(theta_raw), kernel, mode="same")
        theta_sm = np.arctan2(s, c)
    else:
        theta_sm = theta_raw

    theta = float(theta_sm[ti])
    dx = float(np.cos(theta))
    dy = float(np.sin(theta))

    # -------------------------
    # Escalado flecha
    # -------------------------
    cS1, cS2 = st.columns(2)
    with cS1:
        scale_mode = st.selectbox(
            "Escalar flecha por",
            ["constante", "|k|", "speed_proxy"],
            key="tw_scale_mode",
        )
    with cS2:
        base_len = st.slider(
            "Longitud base flecha", 0.01, 0.20, 0.06, key="tw_base_len"
        )

    kmag = float(out["k_mag"][ti]) if "k_mag" in out else 1.0
    spd = float(out["speed_proxy"][ti]) if "speed_proxy" in out else 1.0

    if scale_mode == "constante":
        L = float(base_len)
    elif scale_mode == "|k|":
        L = float(base_len) * (kmag / (np.nanmax(out["k_mag"]) + 1e-12))
    else:
        L = float(base_len) * (spd / (np.nanmax(out["speed_proxy"]) + 1e-12))

    # -------------------------
    # DEBUG (3 líneas): cos/sin vs tiempo
    # -------------------------
    st.caption("DEBUG circular: cos(θ) y sin(θ) (mejor que θ crudo por el salto en ±π)")
    fig_dbg, ax = plt.subplots(figsize=(6.0, 1.4), dpi=130)
    ax.plot(out["times"], np.cos(theta_sm), label="cos(θ)", lw=1)
    ax.plot(out["times"], np.sin(theta_sm), label="sin(θ)", lw=1)
    ax.axvline(t_vis, color="r", lw=1, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlabel("t (s)", fontsize=9)
    ax.set_ylabel("valor", fontsize=9)
    ax.tick_params(labelsize=8)
    fig_dbg.tight_layout(pad=0.2)
    st.pyplot(fig_dbg)

    # -------------------------
    # Cache de fase (para no recalcular con cada frame)
    # -------------------------
    phi_key = (tuple(used), float(eeg.sfreq), float(lo_used), float(hi_used))
    if st.session_state.get("tw_phi_cache", {}).get("key") != phi_key:
        from src.waves import bandpass, phase_hilbert

        idx_used = [eeg.ch_names.index(ch) for ch in used]
        X_used = eeg.X[:, idx_used]
        Xf = bandpass(X_used, eeg.sfreq, lo_used, hi_used)
        phi = phase_hilbert(Xf)  # (n_samples, n_used)
        st.session_state["tw_phi_cache"] = {"key": phi_key, "phi": phi}
    phi = st.session_state["tw_phi_cache"]["phi"]

    sample_vis = int(np.clip(round(t_vis * eeg.sfreq), 0, phi.shape[0] - 1))
    phi_t = phi[sample_vis, :]
    phi_t = np.angle(np.exp(1j * (phi_t - np.mean(phi_t))))  # centrado circular

    info_u = mne.create_info(
        ch_names=used, sfreq=eeg.sfreq, ch_types=["eeg"] * len(used)
    )
    mont_u = mne.channels.make_dig_montage(
        {ch: coords_used[ch] for ch in used}, coord_frame="head"
    )
    info_u.set_montage(mont_u)

    # -------------------------
    # Trail (UI)
    # -------------------------
    trail_on = st.checkbox("Mostrar trail", value=True, key="tw_trail_on")
    trail_len = st.slider("Longitud trail (frames)", 1, 80, 20, key="tw_trail_len")

    # -------------------------
    # Layout compacto: 2 columnas (topomap + flecha)
    # -------------------------
    c1, c2 = st.columns(2, gap="small")

    with c1:
        st.caption("Fase (topomap)")
        fig_topo, ax = plt.subplots(figsize=(2.2, 2.2), dpi=130)
        try:
            ret = mne.viz.plot_topomap(
                phi_t, info_u, axes=ax, show=False, cmap="RdBu_r", vlim=(-np.pi, np.pi)
            )
        except TypeError:
            ret = mne.viz.plot_topomap(
                phi_t,
                info_u,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vmin=-np.pi,
                vmax=np.pi,
            )
        im = ret[0] if isinstance(ret, tuple) else ret
        fig_topo.colorbar(im, ax=ax, shrink=0.55)
        ax.set_title(f"t={t_vis:.1f}s  {band_name_used}", fontsize=9)
        fig_topo.tight_layout(pad=0.3)
        st.pyplot(fig_topo)

    with c2:
        st.caption("Vector (propagación)")
        fig_vec, axv = plt.subplots(figsize=(2.4, 2.4), dpi=130)
        axv.scatter(xy[:, 0], xy[:, 1], s=14)

        for (x, y_), ch in zip(xy, used):
            axv.text(x, y_, ch, fontsize=7, ha="center", va="center")

        if trail_on:
            i0 = max(0, ti - int(trail_len) + 1)
            idxs = np.arange(i0, ti + 1)
            tips_x = cx + np.cos(theta_sm[idxs]) * L
            tips_y = cy + np.sin(theta_sm[idxs]) * L
            axv.plot(tips_x, tips_y, lw=1)

        axv.quiver(
            cx,
            cy,
            dx * L,
            dy * L,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.010,
        )
        axv.set_aspect("equal")
        axv.axis("off")
        axv.set_title(f"θ={theta:.2f} rad", fontsize=9)
        fig_vec.tight_layout(pad=0.2)
        st.pyplot(fig_vec)

    # -------------------------
    # Serie temporal dirección (compacta)  ✅ con placeholders
    # -------------------------
    ph_dir_caption = st.empty()
    ph_dir_plot = st.empty()
    ph_metrics = st.empty()

    ph_dir_caption.caption("Dirección (θ) en el tiempo — línea roja = instante actual")

    fig_dir, ax = plt.subplots(figsize=(6.0, 1.6), dpi=130)
    ax.plot(out["times"], theta_sm, lw=1)
    ax.axvline(t_vis, color="r", lw=1, alpha=0.6)
    ax.set_xlabel("t (s)", fontsize=9)
    ax.set_ylabel("θ (rad)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig_dir.tight_layout(pad=0.3)

    ph_dir_caption.caption("Dirección (θ) en el tiempo — línea roja = instante actual")

    fig_dir, ax = plt.subplots(figsize=(6.0, 1.6), dpi=130)
    ax.plot(out["times"], theta_sm, lw=1)
    ax.axvline(t_vis, color="r", lw=1, alpha=0.6)
    ax.set_xlabel("t (s)", fontsize=9)
    ax.set_ylabel("θ (rad)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig_dir.tight_layout(pad=0.3)
    ph_dir_plot.pyplot(fig_dir)
    plt.close(fig_dir)  # ✅ importante con rerun/play

    ph_metrics.caption(
        f"t={t_vis:.2f}s | θ={theta:.2f} rad | |k|={kmag:.3g} | speed={spd:.3g}"
    )

    # -------------------------
    # Export GIF (vector + trail)
    # -------------------------
    st.subheader("Exportar animación (GIF)")

    colG1, colG2, colG3 = st.columns(3)
    with colG1:
        gif_stride = st.slider("Stride (frames)", 1, 10, 2, key="tw_gif_stride")
    with colG2:
        gif_fps = st.slider("FPS GIF", 1, 20, 8, key="tw_gif_fps")
    with colG3:
        gif_trail = st.slider("Trail en GIF (frames)", 0, 80, 20, key="tw_gif_trail")

    gif_range = st.slider(
        "Rango de frames",
        0,
        nT - 1,
        (max(0, ti - 40), min(nT - 1, ti + 40)),
        key="tw_gif_range",
    )

    if st.button("Generar GIF", key="tw_make_gif"):
        import imageio.v2 as imageio

        start_i, end_i = gif_range
        frames = []

        for jj in range(start_i, end_i + 1, int(gif_stride)):
            th = float(theta_sm[jj])
            dxx, dyy = float(np.cos(th)), float(np.sin(th))

            fig, axg = plt.subplots(figsize=(2.6, 2.6), dpi=120)
            axg.scatter(xy[:, 0], xy[:, 1], s=12)

            for (x, y_), ch in zip(xy, used):
                axg.text(x, y_, ch, fontsize=6, ha="center", va="center")

            if gif_trail > 0:
                i0 = max(0, jj - int(gif_trail) + 1)
                idxs = np.arange(i0, jj + 1)
                tips_x = cx + np.cos(theta_sm[idxs]) * L
                tips_y = cy + np.sin(theta_sm[idxs]) * L
                axg.plot(tips_x, tips_y, lw=1)

            axg.quiver(
                cx,
                cy,
                dxx * L,
                dyy * L,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.010,
            )
            axg.set_aspect("equal")
            axg.axis("off")
            axg.set_title(f"frame {jj}  θ={th:.2f}", fontsize=8)
            fig.tight_layout(pad=0.2)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                h, w, 3
            )
            frames.append(img)
            plt.close(fig)

        buf = io.BytesIO()
        imageio.mimsave(buf, frames, format="GIF", fps=int(gif_fps))

        st.download_button(
            "Descargar GIF",
            data=buf.getvalue(),
            file_name=f"tw_vector_{band_name_used}_{start_i}-{end_i}.gif",
            mime="image/gif",
        )
        st.success("GIF generado ✅")
