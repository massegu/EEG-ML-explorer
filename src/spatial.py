# src/spatial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

DEFAULT_GROUPS: Dict[str, Set[str]] = {
    "FRONTAL": {"Fp1","Fp2","F3","F4","F7","F8","Fz"},
    "POSTERIOR": {"P3","P4","P7","P8","O1","O2","Pz","Oz"},
    "LEFT": {"Fp1","F3","F7","C3","T7","P3","P7","O1"},
    "RIGHT": {"Fp2","F4","F8","C4","T8","P4","P8","O2"},
}

def _idx_set(ch_names: Sequence[str], S: Set[str]) -> List[int]:
    return [i for i, ch in enumerate(ch_names) if ch in S]

def spatial_features_from_X_feat(
    X_feat: np.ndarray,
    ch_names: List[str],
    band_names: List[str],
    target_band: str = "alpha",
    coords: Optional[Dict[str, np.ndarray]] = None,
    groups: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extrae features espaciales por ventana a partir de las features bandpower (canal×banda)
    contenidas en X_feat.

    Importante: asume que las primeras (n_ch*n_b) columnas de X_feat son bandpower
    en el orden: [ch1_band1, ch1_band2, ... ch2_band1, ...] como genera window_bandpower_features.
    Si después se concatenan TW u otras features, no pasa nada: aquí solo se usa el prefijo.

    Devuelve:
      S: (n_windows, n_spatial_features)
      names: nombres de columnas
    """
    if groups is None:
        groups = DEFAULT_GROUPS

    n_w = X_feat.shape[0]
    n_ch = len(ch_names)
    n_b = len(band_names)

    if n_ch == 0 or n_b == 0:
        return np.zeros((n_w, 0)), []

    # Reshape seguro: solo el bloque bandpower
    need = n_ch * n_b
    if X_feat.shape[1] < need:
        # no hay suficientes columnas para bandpower
        return np.zeros((n_w, 0)), []

    X_bp = X_feat[:, :need]
    X3 = X_bp.reshape((n_w, n_ch, n_b))

    if target_band not in band_names:
        target_band = band_names[0]
    b_idx = band_names.index(target_band)

    band_map = X3[:, :, b_idx]  # (n_windows, n_ch)

    # grupos
    idxF = _idx_set(ch_names, groups["FRONTAL"])
    idxP = _idx_set(ch_names, groups["POSTERIOR"])
    idxL = _idx_set(ch_names, groups["LEFT"])
    idxR = _idx_set(ch_names, groups["RIGHT"])

    feats: List[np.ndarray] = []
    names: List[str] = []

    # AP gradient
    if idxF and idxP:
        ap = np.nanmean(band_map[:, idxP], axis=1) - np.nanmean(band_map[:, idxF], axis=1)
        feats.append(ap)
        names.append(f"sp_ap_post_minus_front_{target_band}")

    # LR asymmetry
    if idxL and idxR:
        lr = np.nanmean(band_map[:, idxL], axis=1) - np.nanmean(band_map[:, idxR], axis=1)
        feats.append(lr)
        names.append(f"sp_lr_left_minus_right_{target_band}")

    # GFP
    gfp = np.nanstd(band_map, axis=1)
    feats.append(gfp)
    names.append(f"sp_gfp_{target_band}")

    # CoM (si coords)
    if coords is not None:
        valid = np.array([ch in coords for ch in ch_names], dtype=bool)
        if int(valid.sum()) >= 4:
            xy = np.array([coords[ch][:2] if ch in coords else (np.nan, np.nan) for ch in ch_names], dtype=float)
            xyv = xy[valid]                 # (n_valid, 2)
            amapv = band_map[:, valid]      # (n_w, n_valid)

            # pesos no negativos (solo para centro de masa visual)
            w = amapv - np.nanmin(amapv, axis=1, keepdims=True)
            denom = np.sum(w, axis=1) + 1e-12
            cx = np.sum(w * xyv[:, 0], axis=1) / denom
            cy = np.sum(w * xyv[:, 1], axis=1) / denom

            feats.append(cx); names.append(f"sp_com_x_{target_band}")
            feats.append(cy); names.append(f"sp_com_y_{target_band}")

    if not feats:
        return np.zeros((n_w, 0)), []

    S = np.column_stack(feats)
    return S.astype(float), names
