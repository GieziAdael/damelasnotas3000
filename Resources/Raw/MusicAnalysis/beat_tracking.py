"""Estimaciones de tempo y pulsos con Librosa y Madmom."""
import os

import librosa
import numpy as np
from madmom.features.beats import (
    RNNBeatProcessor,
    DBNBeatTrackingProcessor,
)

from audio_features import HOP_LENGTH, finite_array

def estimate_librosa(audio_path):
    """Carga el audio y obtiene tempo y pulsos de Librosa."""
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"No se encontró el audio: {audio_path}"
        )

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    tempo_values, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    tempo_values = np.asarray(
        tempo_values,
        dtype=float,
    ).reshape(-1)

    librosa_tempo = None

    if tempo_values.size:
        value = float(tempo_values[0])

        if np.isfinite(value) and value > 0:
            librosa_tempo = value

    beat_frames = np.asarray(
        beat_frames,
        dtype=float,
    ).reshape(-1)

    beat_frames = beat_frames[np.isfinite(beat_frames)]

    librosa_beats = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return (
        librosa_tempo,
        finite_array(librosa_beats),
        y,
        sr,
    )

def estimate_madmom_beats(audio_path):
    """Obtiene los tiempos de pulso usando Madmom."""
    activations = RNNBeatProcessor()(audio_path)

    tracker = DBNBeatTrackingProcessor(
        fps=100,
        min_bpm=50,
        max_bpm=220,
    )

    beat_times = tracker(activations)

    return finite_array(beat_times)

def build_tempo_candidates(librosa_tempo, madmom_tempo):
    """Conserva los tempos medidos y sus alternativas a mitad o al doble."""
    raw_tempos = [
        value
        for value in (librosa_tempo, madmom_tempo)
        if value is not None and np.isfinite(value) and value > 0
    ]

    raw_candidates = []

    for value in raw_tempos:
        raw_candidates.extend(
            (
                value / 2.0,
                value,
                value * 2.0,
            )
        )

    candidates = []

    for value in sorted(raw_candidates):
        if not 30.0 <= value <= 300.0:
            continue

        rounded_value = round(float(value), 1)

        duplicate = any(
            abs(rounded_value - existing) / existing <= 0.02
            for existing in candidates
        )

        if not duplicate:
            candidates.append(rounded_value)

    return candidates

def describe_tempo_relation(librosa_tempo, madmom_tempo):
    """Describe si los tempos son similares o difieren por mitad/doble."""
    if (
        librosa_tempo is None
        or madmom_tempo is None
        or librosa_tempo <= 0
        or madmom_tempo <= 0
    ):
        return {
            "relation": "insufficient_data",
            "madmomToLibrosaRatio": None,
            "absoluteDifferenceBpm": None,
        }

    ratio = madmom_tempo / librosa_tempo
    difference = abs(madmom_tempo - librosa_tempo)

    if 0.88 <= ratio <= 1.12:
        relation = "similar"
    elif 1.75 <= ratio <= 2.25:
        relation = "madmom_approximately_double"
    elif 0.44 <= ratio <= 0.57:
        relation = "madmom_approximately_half"
    else:
        relation = "different"

    return {
        "relation": relation,
        "madmomToLibrosaRatio": float(ratio),
        "absoluteDifferenceBpm": float(difference),
    }
