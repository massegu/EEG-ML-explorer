# src/montage.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

def parse_custom_montage_csv(text: str) -> Dict[str, np.ndarray]:
    """
    Espera líneas: name,x,y,z (en metros). Permite comentarios con #.
    Devuelve dict: ch -> np.array([x,y,z]).
    """
    coords: Dict[str, np.ndarray] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        x = float(parts[1]); y = float(parts[2])
        z = float(parts[3]) if len(parts) >= 4 and parts[3] != "" else 0.0
        coords[name] = np.array([x, y, z], dtype=float)
    return coords

def get_standard_coords(ch_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Intenta asignar coords usando montajes estándar de MNE.
    standard_1005 suele cubrir FC/CP/etc. mejor que standard_1020.
    Devuelve dict ch_original -> coords(x,y,z).
    """
    import mne

    for montage_name in ["standard_1005", "standard_1020"]:
        montage = mne.channels.make_standard_montage(montage_name)
        pos = montage.get_positions()["ch_pos"]  # dict name -> (x,y,z)
        out: Dict[str, np.ndarray] = {}

        for ch in ch_names:
            key = ch.strip().replace(".", "")
            if key in pos:
                out[ch] = np.array(pos[key], dtype=float)
            elif key.upper() in pos:
                out[ch] = np.array(pos[key.upper()], dtype=float)

        if len(out) >= 4:
            return out

    return out
