# src/features.py
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
from scipy.signal import welch

DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

def welch_psd(X: np.ndarray, sfreq: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    psds = []
    freqs = None
    for ch in range(X.shape[1]):
        f, p = welch(X[:, ch], fs=sfreq, nperseg=min(nperseg, X.shape[0]))
        freqs = f if freqs is None else freqs
        psds.append(p)
    return freqs, np.vstack(psds)

def bandpower_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Dict[str, Tuple[float, float]]
) -> Dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        out[name] = psd[:, idx].mean(axis=1) if np.any(idx) else np.zeros(psd.shape[0])
    return out

def pca_channels(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    n_components = max(1, min(int(n_components), X.shape[0], X.shape[1]))
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(Xz)
    return comps, pca.explained_variance_ratio_

def window_bandpower_features(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    bands: Dict[str, Tuple[float, float]],
    win_sec: float = 2.0,
    step_sec: float = 1.0,
    nperseg: int = 512,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n_samples, n_ch = X.shape
    win = max(8, int(round(win_sec * sfreq)))
    step = max(1, int(round(step_sec * sfreq)))

    starts = np.arange(0, max(0, n_samples - win + 1), step, dtype=int)
    band_items = list(bands.items())

    feat_names = [f"{ch}_{bname}" for ch in ch_names for bname, _ in band_items]
    X_feat = np.zeros((len(starts), n_ch * len(band_items)), dtype=float)
    t_centers = np.zeros(len(starts), dtype=float)

    for i, s0 in enumerate(starts):
        seg = X[s0:s0 + win, :]
        t_centers[i] = (s0 + win / 2) / sfreq

        this_nperseg = min(nperseg, seg.shape[0])
        col = 0
        for ch in range(n_ch):
            freqs, pxx = welch(seg[:, ch], fs=sfreq, nperseg=this_nperseg)
            for _, (lo, hi) in band_items:
                idx = (freqs >= lo) & (freqs <= hi)
                X_feat[i, col] = float(np.mean(pxx[idx])) if np.any(idx) else 0.0
                col += 1

    return X_feat, t_centers, feat_names

