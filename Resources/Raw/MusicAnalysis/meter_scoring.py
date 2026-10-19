"""Puntuación de compases mediante acentos, bajo y probabilidades de downbeat."""
import os
import sys
from functools import lru_cache

import librosa
import numpy as np

from audio_features import (
    HOP_LENGTH,
    finite_array,
    robust_normalize,
    sample_onset_strength,
)
from beat_grid import make_even_subdivision_grid

MIN_METER_SCORE_MARGIN = 0.12
PROBABILITY_EPSILON = 1e-6

# Frecuencias que suelen contener fundamentales y primeros armónicos
# de las notas graves del piano.
BASS_MIN_HZ = 35.0
BASS_MAX_HZ = 350.0
BASS_N_FFT = 4096


@lru_cache(maxsize=4)
def load_bass_onset_envelope(audio_path):
    """
    Calcula una envolvente de ataques de baja frecuencia.

    El resultado queda cacheado por ruta para no volver a cargar y analizar
    el mismo audio cada vez que se prueba otra cuadrícula o compás.
    """
    if not audio_path:
        return None

    absolute_path = os.path.abspath(audio_path)

    if not os.path.isfile(absolute_path):
        return None

    try:
        audio, sample_rate = librosa.load(
            absolute_path,
            sr=None,
            mono=True,
        )

        if audio.size < 2:
            return None

        spectrum = librosa.stft(
            audio,
            n_fft=BASS_N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        )

        magnitude = np.abs(spectrum).astype(np.float32)
        del spectrum

        frequencies = librosa.fft_frequencies(
            sr=sample_rate,
            n_fft=BASS_N_FFT,
        )

        bass_mask = (
            (frequencies >= BASS_MIN_HZ)
            & (frequencies <= BASS_MAX_HZ)
        )

        if np.count_nonzero(bass_mask) < 2:
            return None

        # Flujo espectral positivo: solo cuenta aumentos de energía.
        positive_flux = np.maximum(
            np.diff(
                magnitude,
                axis=1,
                prepend=magnitude[:, :1],
            ),
            0.0,
        )

        bass_flux = np.sum(
            positive_flux[bass_mask, :],
            axis=0,
        )

        if (
            bass_flux.size < 8
            or not np.all(np.isfinite(bass_flux))
            or float(np.std(bass_flux)) < 1e-10
        ):
            return None

        bass_envelope = robust_normalize(bass_flux)

        if float(np.std(bass_envelope)) < 1e-6:
            return None

        return (
            int(sample_rate),
            np.asarray(bass_envelope, dtype=np.float32),
        )

    except Exception as exc:
        print(
            "AVISO: no se pudo calcular la envolvente grave: "
            f"{exc}",
            file=sys.stderr,
        )
        return None


def sample_bass_onset_strength(audio_path, grid_times):
    """Muestrea ataques de baja frecuencia en los tiempos de una cuadrícula."""
    cached = load_bass_onset_envelope(audio_path)

    if cached is None:
        return None

    sample_rate, bass_envelope = cached

    return sample_onset_strength(
        bass_envelope,
        sample_rate,
        grid_times,
    )


def align_downbeat_probabilities(bar_activations, beat_times):
    """Alinea las probabilidades de downbeat con la cuadrícula recibida."""
    activations = np.asarray(
        bar_activations,
        dtype=float,
    )
    beat_times = finite_array(beat_times)

    if (
        activations.ndim != 2
        or activations.shape[1] < 2
        or not len(beat_times)
    ):
        return None

    probabilities = activations[:, 1]

    if not np.any(np.isfinite(probabilities)):
        return None

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    activation_times = activations[:, 0]

    if (
        len(activation_times) == len(probabilities)
        and len(activation_times) >= 2
        and np.all(np.isfinite(activation_times))
        and np.all(np.diff(activation_times) > 0)
    ):
        aligned = np.interp(
            beat_times,
            activation_times,
            probabilities,
        )
        return np.clip(aligned, 0.0, 1.0)

    if len(probabilities) == len(beat_times):
        return probabilities

    if len(probabilities) >= 2:
        old_positions = np.linspace(
            0.0,
            1.0,
            len(probabilities),
        )
        new_positions = np.linspace(
            0.0,
            1.0,
            len(beat_times),
        )
        aligned = np.interp(
            new_positions,
            old_positions,
            probabilities,
        )
        return np.clip(aligned, 0.0, 1.0)

    return None


def get_downbeat_probabilities(audio_path, beat_times, bar_processor):
    """Obtiene probabilidades de downbeat usando una cuadrícula dada."""
    if len(beat_times) < 4 or bar_processor is None:
        return None

    try:
        activations = bar_processor(
            (audio_path, beat_times)
        )

        return align_downbeat_probabilities(
            activations,
            beat_times,
        )
    except Exception as exc:
        print(
            "AVISO: no se pudieron calcular downbeats "
            f"para esta cuadrícula: {exc}",
            file=sys.stderr,
        )
        return None


def score_downbeat_period(probabilities, period, phase_index=None):
    """
    Mide si la probabilidad de downbeat se repite en la misma posición
    de cada compás. Requiere al menos dos compases completos.
    """
    if probabilities is None or period < 2:
        return None

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(probabilities) < period * 2:
        return None

    if not np.any(np.isfinite(probabilities)):
        return None

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    probabilities = np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    complete_bars = len(probabilities) // period

    if complete_bars < 2:
        return None

    bar_values = probabilities[
        :complete_bars * period
    ].reshape(complete_bars, period)

    phase_scores = []
    phases = (
        range(period)
        if phase_index is None
        else (int(phase_index) % period,)
    )

    for phase in phases:
        downbeat_values = bar_values[:, phase]
        other_values = np.delete(
            bar_values,
            phase,
            axis=1,
        ).reshape(-1)

        if not len(other_values):
            continue

        contrast = (
            float(np.median(downbeat_values))
            - float(np.median(other_values))
        )

        per_bar_other = np.delete(
            bar_values,
            phase,
            axis=1,
        )
        per_bar_contrast = (
            downbeat_values
            - np.median(per_bar_other, axis=1)
        )

        consistency = float(
            np.mean(per_bar_contrast > 0.0)
        )

        # Contraste expresa cuánto sobresale el downbeat; consistencia
        # evita que un único compás atípico decida toda la puntuación.
        score = (
            0.75 * contrast
            + 0.25 * (2.0 * consistency - 1.0)
        )

        if np.isfinite(score):
            phase_scores.append(float(score))

    return max(phase_scores) if phase_scores else None


def _score_accent_pattern_diagnostic(
    features,
    pattern,
    phase_index=None,
):
    """
    Calcula el ajuste del patrón e indica por qué no pudo puntuarlo.

    Devuelve (puntuación, motivo, diagnóstico). phase_index permite que
    varias fuentes se evalúen con la misma fase de compás.
    """
    features = np.asarray(
        features,
        dtype=float,
    ).reshape(-1)
    pattern = np.asarray(
        pattern,
        dtype=float,
    ).reshape(-1)

    period = len(pattern)
    diagnostics = {
        "sampleCount": int(len(features)),
        "finiteSampleCount": 0,
        "period": int(period),
        "barCount": 0,
        "featureStd": None,
        "profileStd": None,
    }

    if period < 2 or not np.all(np.isfinite(pattern)):
        return None, "invalid_pattern", diagnostics

    finite_mask = np.isfinite(features)
    finite_count = int(np.count_nonzero(finite_mask))
    diagnostics["finiteSampleCount"] = finite_count

    if finite_count < period * 2:
        return None, "insufficient_samples", diagnostics

    finite_mean = float(np.mean(features[finite_mask]))
    features = np.where(
        finite_mask,
        features,
        finite_mean,
    )

    complete_bars = len(features) // period
    diagnostics["barCount"] = int(complete_bars)
    diagnostics["featureStd"] = float(np.std(features))

    if complete_bars < 2:
        return None, "insufficient_complete_bars", diagnostics

    bar_values = features[
        :complete_bars * period
    ].reshape(complete_bars, period)

    position_profile = np.median(bar_values, axis=0)
    profile_centered = (
        position_profile - float(np.mean(position_profile))
    )
    profile_scale = float(np.std(profile_centered))
    diagnostics["profileStd"] = profile_scale

    if not np.isfinite(profile_scale) or profile_scale < 1e-8:
        return None, "flat_position_profile", diagnostics

    normalized_bars = (
        bar_values - np.mean(
            bar_values,
            axis=1,
            keepdims=True,
        )
    )
    bar_scales = np.std(
        normalized_bars,
        axis=1,
        keepdims=True,
    )

    valid_bar_mask = bar_scales[:, 0] >= 1e-8

    normalized_bars = np.divide(
        normalized_bars,
        bar_scales,
        out=np.zeros_like(normalized_bars),
        where=bar_scales >= 1e-8,
    )

    best_score = None

    phases = (
        range(period)
        if phase_index is None
        else (int(phase_index) % period,)
    )

    for phase in phases:
        expected = pattern[
            (np.arange(period) - phase) % period
        ]
        expected_centered = (
            expected - float(np.mean(expected))
        )
        expected_scale = float(np.std(expected_centered))

        if (
            not np.isfinite(expected_scale)
            or expected_scale < 1e-8
        ):
            continue

        profile_score = float(
            np.mean(profile_centered * expected_centered)
            / (profile_scale * expected_scale)
        )

        if np.any(valid_bar_mask):
            expected_unit = expected_centered / expected_scale
            bar_scores = np.mean(
                normalized_bars[valid_bar_mask] * expected_unit,
                axis=1,
            )
            consistency_score = float(np.median(bar_scores))
        else:
            consistency_score = profile_score

        score = (
            0.75 * profile_score
            + 0.25 * consistency_score
        )

        if np.isfinite(score) and (
            best_score is None or score > best_score
        ):
            best_score = float(score)

    if best_score is None:
        return None, "no_finite_phase_score", diagnostics

    return best_score, None, diagnostics


def score_accent_pattern(features, pattern, phase_index=None):
    """Puntúa un patrón, opcionalmente en una fase de compás compartida."""
    score, _, _ = _score_accent_pattern_diagnostic(
        features,
        pattern,
        phase_index=phase_index,
    )
    return score


def score_meter_components(
    beat_features,
    subdivision_features,
    downbeat_probabilities,
    downbeat_period,
    accent_pattern,
    use_subdivision_grid=False,
    hierarchy_features=None,
    hierarchy_pattern=None,
    bass_features=None,
    hierarchy_bass_features=None,
    subdivisions_per_pulse=1,
):
    """Combina evidencias usando una fase común para todo el compás."""
    if use_subdivision_grid:
        accent_input = subdivision_features
    else:
        accent_input = beat_features

    phase_count = max(1, int(downbeat_period))
    subdivisions_per_pulse = max(1, int(subdivisions_per_pulse))
    phase_results = []
    first_accent_diagnostics = None
    first_accent_reason = None

    for beat_phase in range(phase_count):
        accent_score, accent_reason, accent_diagnostics = (
            _score_accent_pattern_diagnostic(
                accent_input,
                accent_pattern,
                phase_index=beat_phase,
            )
        )

        if first_accent_diagnostics is None:
            first_accent_diagnostics = accent_diagnostics
            first_accent_reason = accent_reason

        downbeat_score = score_downbeat_period(
            downbeat_probabilities,
            downbeat_period,
            phase_index=beat_phase,
        )

        hierarchy_phase = (
            beat_phase * subdivisions_per_pulse
        )
        hierarchy_score = None

        if hierarchy_features is not None and hierarchy_pattern is not None:
            hierarchy_score = score_accent_pattern(
                hierarchy_features,
                hierarchy_pattern,
                phase_index=hierarchy_phase,
            )

        bass_scores = []

        if bass_features is not None:
            value = score_accent_pattern(
                bass_features,
                accent_pattern,
                phase_index=beat_phase,
            )

            if value is not None and np.isfinite(value):
                bass_scores.append(float(value))

        if (
            hierarchy_bass_features is not None
            and hierarchy_pattern is not None
        ):
            value = score_accent_pattern(
                hierarchy_bass_features,
                hierarchy_pattern,
                phase_index=hierarchy_phase,
            )

            if value is not None and np.isfinite(value):
                bass_scores.append(float(value))

        bass_score = (
            float(np.median(bass_scores))
            if bass_scores
            else None
        )

        weighted_components = [
            ("onsetAccentScore", 0.30, accent_score),
            ("hierarchyScore", 0.25, hierarchy_score),
            ("bassAccentScore", 0.25, bass_score),
            ("downbeatScore", 0.20, downbeat_score),
        ]

        available = [
            (name, weight, float(value))
            for name, weight, value in weighted_components
            if value is not None and np.isfinite(value)
        ]

        details = {
            name: (
                float(value)
                if value is not None and np.isfinite(value)
                else None
            )
            for name, _, value in weighted_components
        }
        details["phaseIndex"] = int(beat_phase)
        details["onsetAccentReason"] = accent_reason
        details["onsetAccentSampleCount"] = accent_diagnostics[
            "finiteSampleCount"
        ]
        details["onsetAccentBarCount"] = accent_diagnostics["barCount"]
        details["onsetAccentProfileStd"] = accent_diagnostics[
            "profileStd"
        ]

        if not available:
            details["score"] = None
            continue

        total_weight = sum(weight for _, weight, _ in available)
        score = sum(
            weight * value
            for _, weight, value in available
        ) / total_weight
        details["score"] = float(score) if np.isfinite(score) else None

        if details["score"] is not None:
            phase_results.append(details)

    if not phase_results:
        diagnostics = first_accent_diagnostics or {}
        return {
            "score": None,
            "onsetAccentScore": None,
            "hierarchyScore": None,
            "bassAccentScore": None,
            "downbeatScore": None,
            "phaseIndex": None,
            "onsetAccentReason": first_accent_reason,
            "onsetAccentSampleCount": diagnostics.get(
                "finiteSampleCount", 0
            ),
            "onsetAccentBarCount": diagnostics.get("barCount", 0),
            "onsetAccentProfileStd": diagnostics.get("profileStd"),
        }

    return max(phase_results, key=lambda item: item["score"])


def score_meter_hypothesis(
    beat_features,
    subdivision_features,
    downbeat_probabilities,
    downbeat_period,
    accent_pattern,
    use_subdivision_grid=False,
    hierarchy_features=None,
    hierarchy_pattern=None,
    bass_features=None,
    hierarchy_bass_features=None,
    subdivisions_per_pulse=1,
):
    """Devuelve la puntuación combinada de una hipótesis de compás."""
    components = score_meter_components(
        beat_features=beat_features,
        subdivision_features=subdivision_features,
        downbeat_probabilities=downbeat_probabilities,
        downbeat_period=downbeat_period,
        accent_pattern=accent_pattern,
        use_subdivision_grid=use_subdivision_grid,
        hierarchy_features=hierarchy_features,
        hierarchy_pattern=hierarchy_pattern,
        bass_features=bass_features,
        hierarchy_bass_features=hierarchy_bass_features,
        subdivisions_per_pulse=subdivisions_per_pulse,
    )


    score = components["score"]

    return components["score"]


METER_PULSE_VARIANTS = {
    (2, 4): [
        {
            "beatsPerBar": 2,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.22],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [1.0, 0.08, 0.28, 0.08],
        },
        {
            "beatsPerBar": 4,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.08, 0.28, 0.08],
            "subdivisionsPerPulse": 1,
        },
    ],
    (3, 8): [
        {
            "beatsPerBar": 3,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.16, 0.34],
            "subdivisionsPerPulse": 1,
        },
    ],
    (3, 4): [
        {
            "beatsPerBar": 3,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.18, 0.34],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.34, 0.08, 0.18, 0.08,
            ],
        },
        {
            "beatsPerBar": 6,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.08, 0.34, 0.08, 0.18, 0.08],
            "subdivisionsPerPulse": 1,
        },
    ],
    (4, 4): [
        {
            "beatsPerBar": 2,
            "beatUnit": "half",
            "beatUnitQuarterLength": 2.0,
            "accentPattern": [1.0, 0.45],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.55, 0.08,
            ],
        },
        {
            "beatsPerBar": 4,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.18, 0.58, 0.15],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.18, 0.08,
                0.58, 0.08, 0.15, 0.08,
            ],
        },
        {
            "beatsPerBar": 8,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [
                1.0, 0.08, 0.18, 0.08,
                0.58, 0.08, 0.15, 0.08,
            ],
            "subdivisionsPerPulse": 1,
        },
    ],
    (5, 4): [
        {
            "beatsPerBar": 5,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.16, 0.48, 0.16, 0.30],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.16, 0.08, 0.48, 0.08,
                0.16, 0.08, 0.30, 0.08,
            ],
        },
        {
            "beatsPerBar": 10,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [
                1.0, 0.08, 0.16, 0.08, 0.48, 0.08,
                0.16, 0.08, 0.30, 0.08,
            ],
            "subdivisionsPerPulse": 1,
        },
    ],
    (5, 8): [
        {
            "beatsPerBar": 5,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.16, 0.48, 0.16, 0.30],
            "subdivisionsPerPulse": 1,
        },
    ],
    (6, 4): [
        {
            "beatsPerBar": 6,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.16, 0.42, 0.18, 0.54, 0.18],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.16, 0.08, 0.42, 0.08,
                0.18, 0.08, 0.54, 0.08, 0.18, 0.08,
            ],
        },
    ],
    (6, 8): [
        {
            "beatsPerBar": 2,
            "beatUnit": "dotted_quarter",
            "beatUnitQuarterLength": 1.5,
            "accentPattern": [1.0, 0.55],
            "subdivisionsPerPulse": 3,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.22, 0.58, 0.08, 0.22,
            ],
        },
        {
            "beatsPerBar": 6,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [
                1.0, 0.08, 0.22, 0.58, 0.08, 0.22,
            ],
            "subdivisionsPerPulse": 1,
        },
    ],
    (7, 4): [
        {
            "beatsPerBar": 7,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.16, 0.48, 0.16, 0.42, 0.16, 0.28],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.16, 0.08, 0.48, 0.08, 0.16, 0.08,
                0.42, 0.08, 0.16, 0.08, 0.28, 0.08,
            ],
        },
    ],
    (7, 8): [
        {
            "beatsPerBar": 7,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.16, 0.48, 0.16, 0.42, 0.16, 0.28],
            "subdivisionsPerPulse": 1,
        },
    ],
    (9, 8): [
        {
            "beatsPerBar": 3,
            "beatUnit": "dotted_quarter",
            "beatUnitQuarterLength": 1.5,
            "accentPattern": [1.0, 0.50, 0.28],
            "subdivisionsPerPulse": 3,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.22, 0.50, 0.08, 0.22,
                0.28, 0.08, 0.22,
            ],
        },
        {
            "beatsPerBar": 9,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [
                1.0, 0.08, 0.22, 0.50, 0.08, 0.22,
                0.28, 0.08, 0.22,
            ],
            "subdivisionsPerPulse": 1,
        },
    ],
    (12, 8): [
        {
            "beatsPerBar": 4,
            "beatUnit": "dotted_quarter",
            "beatUnitQuarterLength": 1.5,
            "accentPattern": [1.0, 0.45, 0.22, 0.55],
            "subdivisionsPerPulse": 3,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.22, 0.45, 0.08, 0.22,
                0.22, 0.08, 0.22, 0.55, 0.08, 0.22,
            ],
        },
        {
            "beatsPerBar": 12,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [
                1.0, 0.08, 0.22, 0.45, 0.08, 0.22,
                0.22, 0.08, 0.22, 0.55, 0.08, 0.22,
            ],
            "subdivisionsPerPulse": 1,
        },
    ],
}


def estimate_meter_for_grid(
    audio_path,
    beat_times,
    onset_envelope,
    sr,
    bar_processor,
):
    """
    Evalúa cada compás con varias unidades posibles.

    Usa tres fuentes de información:
    - acentos generales de audio;
    - ataques de baja frecuencia, que suelen resaltar el bajo;
    - probabilidades de downbeat de Madmom.

    Los campos de puntuación por fuente se incluyen en cada candidato
    para facilitar la comparación en el log.
    """
    beat_times = finite_array(beat_times)

    empty_result = {
        "selected": None,
        "margin": None,
        "candidates": [],
        "pulseHypotheses": [],
    }

    if len(beat_times) < 8:
        return empty_result

    beat_features = sample_onset_strength(
        onset_envelope,
        sr,
        beat_times,
    )

    bass_beat_features = sample_bass_onset_strength(
        audio_path,
        beat_times,
    )

    downbeat_probabilities = get_downbeat_probabilities(
        audio_path,
        beat_times,
        bar_processor,
    )

    pulse_hypotheses = []
    best_by_meter = {}

    for (numerator, denominator), variants in (
        METER_PULSE_VARIANTS.items()
    ):
        for variant in variants:
            pulses_per_bar = int(variant["beatsPerBar"])
            subdivisions_per_pulse = int(
                variant.get("subdivisionsPerPulse", 1)
            )

            if len(beat_features) < pulses_per_bar * 2:
                continue

            hierarchy_features = None
            hierarchy_bass_features = None
            hierarchy_pattern = variant.get(
                "subdivisionAccentPattern"
            )

            if (
                subdivisions_per_pulse > 1
                and hierarchy_pattern is not None
            ):
                hierarchy_grid = make_even_subdivision_grid(
                    beat_times,
                    subdivisions_per_pulse,
                )

                hierarchy_features = sample_onset_strength(
                    onset_envelope,
                    sr,
                    hierarchy_grid,
                )

                hierarchy_bass_features = sample_bass_onset_strength(
                    audio_path,
                    hierarchy_grid,
                )

            components = score_meter_components(
                beat_features=beat_features,
                subdivision_features=beat_features,
                downbeat_probabilities=downbeat_probabilities,
                downbeat_period=pulses_per_bar,
                accent_pattern=variant["accentPattern"],
                hierarchy_features=hierarchy_features,
                hierarchy_pattern=hierarchy_pattern,
                bass_features=bass_beat_features,
                hierarchy_bass_features=hierarchy_bass_features,
                subdivisions_per_pulse=subdivisions_per_pulse,
            )

            print(
                "DEBUG onsetAccentScore:",
                repr(components["onsetAccentScore"]),
                "reason:",
                components["onsetAccentReason"],
                "profileStd:",
                repr(components["onsetAccentProfileStd"]),
                "meter:",
                f"{numerator}/{denominator}",
                "unit:",
                variant["beatUnit"],
                "phase:",
                components["phaseIndex"],
            )

            score = components["score"]

            if score is None or not np.isfinite(score):
                continue

            candidate = {
                "numerator": numerator,
                "denominator": denominator,
                "beatsPerBar": pulses_per_bar,
                "beatUnit": variant["beatUnit"],
                "beatUnitQuarterLength": float(
                    variant["beatUnitQuarterLength"]
                ),
                "subdivisionsPerPulse": subdivisions_per_pulse,
                "score": float(score),
                "onsetAccentScore":
                    components["onsetAccentScore"],
                "onsetAccentReason":
                    components["onsetAccentReason"],
                "onsetAccentProfileStd":
                    components["onsetAccentProfileStd"],
                "phaseIndex":
                    components["phaseIndex"],
                "hierarchyScore":
                    components["hierarchyScore"],
                "bassAccentScore":
                    components["bassAccentScore"],
                "downbeatScore":
                    components["downbeatScore"],
            }

            pulse_hypotheses.append(candidate)

            meter_key = (numerator, denominator)
            current = best_by_meter.get(meter_key)

            if (
                current is None
                or candidate["score"] > current["score"]
            ):
                best_by_meter[meter_key] = candidate

    pulse_hypotheses.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    candidates = list(best_by_meter.values())
    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    if len(candidates) < 2:
        return {
            "selected": None,
            "margin": None,
            "candidates": candidates,
            "pulseHypotheses": pulse_hypotheses,
        }

    margin = float(
        candidates[0]["score"] - candidates[1]["score"]
    )

    selected = None

    if margin >= MIN_METER_SCORE_MARGIN:
        selected = {
            "numerator": candidates[0]["numerator"],
            "denominator": candidates[0]["denominator"],
            "beatsPerBar": candidates[0]["beatsPerBar"],
            "beatUnit": candidates[0]["beatUnit"],
            "beatUnitQuarterLength":
                candidates[0]["beatUnitQuarterLength"],
        }

    return {
        "selected": selected,
        "margin": margin,
        "candidates": candidates,
        "pulseHypotheses": pulse_hypotheses,
    }
