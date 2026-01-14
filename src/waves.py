# src/waves.py
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

def bandpass(X: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sfreq
    b, a = butter(4, [lo/nyq, hi/nyq], btype="bandpass")
    return filtfilt(b, a, X, axis=0)

def phase_hilbert(X: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert
    analytic = hilbert(X, axis=0)
    return np.angle(analytic)

def unwrap_phase(phi: np.ndarray) -> np.ndarray:
    # unwrap por tiempo
    return np.unwrap(phi, axis=0)

def plane_fit_k(phi_t: np.ndarray, xy: np.ndarray) -> Tuple[float, float, float]:
    """
    Ajusta phi ≈ kx*x + ky*y + c por mínimos cuadrados en un instante t.
    phi_t: (n_channels,)
    xy: (n_channels, 2)
    """
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(xy.shape[0])])
    kx, ky, c = np.linalg.lstsq(A, phi_t, rcond=None)[0]
    return float(kx), float(ky), float(c)

def traveling_wave_metrics(
    X: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    coords: Dict[str, np.ndarray],
    band: Tuple[float, float],
    t0: float,
    t1: float
) -> Dict[str, np.ndarray]:
    """
    Devuelve series temporales:
      - direction_rad (n_times,)
      - k_mag (n_times,)
      - speed_proxy (n_times,)
      - times (n_times,)
    """
    lo, hi = band
    # seleccionar canales con coords
    idx = [i for i, ch in enumerate(ch_names) if ch in coords]
    if len(idx) < 4:
        raise ValueError("Necesitas al menos 4 canales con coordenadas para ajustar traveling waves.")

    Xc = X[:, idx]
    chs = [ch_names[i] for i in idx]
    xy = np.array([coords[ch][:2] for ch in chs], dtype=float)

    # recorte temporal
    n = X.shape[0]
    times = np.arange(n) / sfreq
    i0 = int(max(0, np.floor(t0 * sfreq)))
    i1 = int(min(n, np.ceil(t1 * sfreq)))
    Xw = Xc[i0:i1]
    tw = times[i0:i1]

    Xf = bandpass(Xw, sfreq, lo, hi)
    phi = phase_hilbert(Xf)
    phi_u = unwrap_phase(phi)

    direction = np.zeros(phi_u.shape[0], dtype=float)
    k_mag = np.zeros(phi_u.shape[0], dtype=float)
    kx_arr = np.zeros(phi_u.shape[0], dtype=float)
    ky_arr = np.zeros(phi_u.shape[0], dtype=float)

    for t in range(phi_u.shape[0]):
        kx, ky, _ = plane_fit_k(phi_u[t], xy)
        kx_arr[t] = kx
        ky_arr[t] = ky
        direction[t] = np.arctan2(ky, kx)
        k_mag[t] = np.sqrt(kx*kx + ky*ky)


    # proxy simple de velocidad (cuanto cambia la dirección y/o k con el tiempo)
    # (si quieres velocidad física real, hay que definir bien unidades y estimación)
    dt = 1.0 / sfreq
    speed_proxy = np.concatenate([[0.0], np.abs(np.diff(k_mag)) / dt])

    return {
        "times": tw,
        "direction_rad": direction,
        "k_mag": k_mag,
        "speed_proxy": speed_proxy,
        "used_channels": np.array(chs, dtype=object),
        "kx": kx_arr,
        "ky": ky_arr,

    }
