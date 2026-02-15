# src/eeg_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union
import numpy as np
import os
import tempfile
import io

import mne


@dataclass
class EEGData:
    X: np.ndarray          # (n_samples, n_channels)
    sfreq: float
    ch_names: List[str]
    times: np.ndarray      # (n_samples,)


def _normalize_ch_name(name: str) -> str:
    s = name.strip().replace("EEG", "").replace("eeg", "").strip()
    s = s.replace(".", "")  # quita puntos tipo Fc1.
    for sep in ["-", "_"]:
        s = s.split(sep)[0].strip()
    # Si queda algo como "Fp1 REF", quita el "REF"
    s = s.split()[-1] if s.split() else s
    return s


def _postprocess_raw(raw: mne.io.BaseRaw) -> EEGData:
    """Aplica picking EEG, normaliza nombres y devuelve EEGData."""
    raw.pick_types(eeg=True, exclude="bads")

    ch_names = [_normalize_ch_name(ch) for ch in raw.ch_names]
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

    X = raw.get_data().T.astype(np.float32)     # (n_samples, n_channels)
    sfreq = float(raw.info["sfreq"])
    times = raw.times.astype(np.float32)

    return EEGData(X=X, sfreq=sfreq, ch_names=ch_names, times=times)


def load_edf_to_eegdata(uploaded_file: Union[str, os.PathLike, object]) -> EEGData:
    """
    Carga EDF desde:
      - ruta (str / PathLike)
      - UploadedFile de Streamlit (tiene getbuffer()).
    """
    # 1) Si ya es una ruta, úsala tal cual
    if isinstance(uploaded_file, (str, os.PathLike)):
        raw = mne.io.read_raw_edf(str(uploaded_file), preload=True, verbose="ERROR")
        return _postprocess_raw(raw)

    # 2) Si viene de Streamlit (UploadedFile), guárdalo a un .edf temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
        return _postprocess_raw(raw)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_edf_bytes_to_eegdata(
    edf_bytes: bytes,
    crop_on: bool = False,
    crop_sec: float = 60.0,
) -> EEGData:
    """
    Carga un EDF desde bytes (Streamlit-friendly) y devuelve EEGData.
    Usa archivo temporal porque muchas versiones de MNE requieren ruta.
    """
    import mne

    # 1) escribir bytes a un EDF temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(edf_bytes)
        tmp_path = tmp.name

    try:
        # 2) leer con MNE desde ruta
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
        raw.pick_types(eeg=True, exclude="bads")

        # 3) normaliza nombres
        ch_names = [_normalize_ch_name(ch) for ch in raw.ch_names]
        raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

        # 4) recorte
        if crop_on:
            tmax = min(float(crop_sec), float(raw.times[-1]))
            raw = raw.copy().crop(tmin=0.0, tmax=tmax)

        # 5) extraer a EEGData
        X = raw.get_data().T.astype(np.float32)
        sfreq = float(raw.info["sfreq"])
        times = raw.times.astype(np.float32)

        return EEGData(X=X, sfreq=sfreq, ch_names=ch_names, times=times)

    finally:
        # 6) borrar temporal sí o sí
        try:
            os.remove(tmp_path)
        except OSError:
            pass



