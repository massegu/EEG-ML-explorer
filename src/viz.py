# src/viz.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

def plot_topomap_band(
    values: np.ndarray,
    chs: List[str],
    coords: Dict[str, np.ndarray],
    sfreq: float,
    title: str = "",
    cmap: str = "viridis",
    vlim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (3.0, 3.0),
    dpi: int = 140,
    show_names: bool = True,
):
    import mne

    info = mne.create_info(ch_names=chs, sfreq=sfreq, ch_types=["eeg"] * len(chs))
    mont = mne.channels.make_dig_montage({ch: coords[ch] for ch in chs}, coord_frame="head")
    info.set_montage(mont)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if vlim is None:
        ret = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap)
    else:
        # compatibilidad MNE antigua: vlim vs vmin/vmax
        try:
            ret = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=vlim)
        except TypeError:
            ret = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vmin=vlim[0], vmax=vlim[1])

    im = ret[0] if isinstance(ret, tuple) else ret
    fig.colorbar(im, ax=ax, shrink=0.75)

    if show_names:
        pos = np.array([coords[ch][:2] for ch in chs], dtype=float)
        for (x, y), ch in zip(pos, chs):
            ax.text(x, y, ch, fontsize=7, ha="center", va="center")

    ax.set_title(title)
    fig.tight_layout()
    return fig
