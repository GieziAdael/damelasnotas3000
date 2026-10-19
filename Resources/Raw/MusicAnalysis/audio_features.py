"""Funciones comunes para cargar y resumir características del audio."""
import librosa
import numpy as np

HOP_LENGTH = 512

def finite_array(values):
    """Devuelve un vector float con entradas no finitas eliminadas."""
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    return values[np.isfinite(values)]

def robust_normalize(values):
    """Normaliza una serie ignorando valores no finitos."""
    raw = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    normalized = np.zeros_like(raw)
    finite_mask = np.isfinite(raw)

    if not np.any(finite_mask):
        return normalized

    finite_values = raw[finite_mask]
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(finite_values))

    if not np.isfinite(scale) or scale < 1e-8:
        return normalized

    normalized[finite_mask] = (
        finite_values - median
    ) / scale

    return np.clip(
        normalized,
        -4.0,
        4.0,
    )

def build_onset_envelope(y, sr):
    """Calcula una sola vez la envolvente de ataques del audio."""
    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return robust_normalize(onset_envelope)

def sample_onset_strength(onset_envelope, sr, grid_times):
    """Muestrea la fuerza de ataque del audio cerca de cada pulso."""
    grid_times = np.asarray(
        grid_times,
        dtype=float,
    ).reshape(-1)

    if not len(grid_times) or not len(onset_envelope):
        return np.zeros(
            len(grid_times),
            dtype=float,
        )

    radius = max(
        1,
        int(round(0.06 * sr / HOP_LENGTH)),
    )

    strengths = np.zeros(
        len(grid_times),
        dtype=float,
    )

    for index, time_seconds in enumerate(grid_times):
        if not np.isfinite(time_seconds):
            continue

        frame = int(
            round(time_seconds * sr / HOP_LENGTH)
        )
        start = max(0, frame - radius)
        end = min(
            len(onset_envelope),
            frame + radius + 1,
        )

        if start < end:
            strengths[index] = float(
                np.max(onset_envelope[start:end])
            )

    return robust_normalize(strengths)
