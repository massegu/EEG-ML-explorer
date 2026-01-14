# src/eeg_io.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import os
import tempfile

@dataclass
class EEGData:
    X: np.ndarray          # (n_samples, n_channels)
    sfreq: float
    ch_names: List[str]
    times: np.ndarray      # (n_samples,)

def _normalize_ch_name(name: str) -> str:
    s = name.strip().replace("EEG", "").replace("eeg", "").strip()
    s = s.replace(".", "")  # ✅ quita puntos tipo Fc1.
    for sep in ["-", "_"]:
        s = s.split(sep)[0].strip()
        # Si queda algo como "Fp1 REF", quita el "REF"
    s = s.split()[-1] if s.split() else s
    return s

def load_edf_to_eegdata(uploaded_file) -> EEGData:
    import mne

    # 1) Si ya es una ruta, úsala tal cual
    if isinstance(uploaded_file, (str, os.PathLike)):
        raw = mne.io.read_raw_edf(str(uploaded_file), preload=True, verbose="ERROR")
    else:
        # 2) Si viene de Streamlit (UploadedFile), guárdalo a un .edf temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    raw.pick_types(eeg=True, exclude="bads")

    ch_names = [_normalize_ch_name(ch) for ch in raw.ch_names]
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

    X = raw.get_data().T
    sfreq = float(raw.info["sfreq"])
    times = raw.times.copy()

    return EEGData(X=X, sfreq=sfreq, ch_names=ch_names, times=times)

