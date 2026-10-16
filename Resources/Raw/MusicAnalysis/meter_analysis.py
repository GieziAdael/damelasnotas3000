import json
import os
import sys

import librosa
import numpy as np

from madmom.features.beats import (
    RNNBeatProcessor,
    DBNBeatTrackingProcessor,
)
from madmom.features.downbeats import RNNBarProcessor


AUDIO_PATH = os.environ.get("AUDIO_INPUT")
MIDI_PATH = os.environ.get("MIDI_INPUT")

HOP_LENGTH = 512
MIN_METER_SCORE_MARGIN = 0.12
PROBABILITY_EPSILON = 1e-6
MIN_TEMPO_HYPOTHESIS_MARGIN = 0.04
MIN_JOINT_METER_MARGIN = 0.04


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


def grid_period(grid):
    """Calcula el intervalo típico entre pulsos."""
    grid = finite_array(grid)

    if len(grid) < 2:
        return None

    intervals = np.diff(grid)
    intervals = intervals[
        np.isfinite(intervals)
        & (intervals > 0)
    ]

    if not intervals.size:
        return None

    return float(np.median(intervals))


def grid_tempo(grid):
    """Calcula BPM desde la cuadrícula de pulsos."""
    period = grid_period(grid)

    if period is None or period <= 0:
        return None

    tempo = 60.0 / period

    if not np.isfinite(tempo):
        return None

    return float(tempo)


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


def compare_beat_grids(librosa_beats, madmom_beats):
    """
    Compara la duración de los pulsos y qué tanto coinciden en el tiempo.

    Si una cuadrícula tiene aproximadamente el doble de pulsos que la otra,
    también prueba alinearla tomando cada segundo pulso.
    """
    librosa_beats = finite_array(librosa_beats)
    madmom_beats = finite_array(madmom_beats)

    librosa_period = grid_period(librosa_beats)
    madmom_period = grid_period(madmom_beats)
    librosa_grid_tempo = grid_tempo(librosa_beats)
    madmom_grid_tempo = grid_tempo(madmom_beats)

    comparison = {
        "librosaGridTempoBpm": librosa_grid_tempo,
        "madmomGridTempoBpm": madmom_grid_tempo,
        "madmomToLibrosaTempoRatio": None,
        "slowerToFasterPulsePeriodRatio": None,
        "expectedFastPulsesPerSlowPulse": None,
        "matchedFraction": None,
        "medianAlignmentErrorMs": None,
        "rateRelation": "insufficient_data",
    }

    if (
        librosa_grid_tempo is not None
        and madmom_grid_tempo is not None
        and librosa_grid_tempo > 0
    ):
        tempo_ratio = madmom_grid_tempo / librosa_grid_tempo
        comparison["madmomToLibrosaTempoRatio"] = float(tempo_ratio)

        if 0.88 <= tempo_ratio <= 1.12:
            comparison["rateRelation"] = "similar"
        elif 1.75 <= tempo_ratio <= 2.25:
            comparison["rateRelation"] = "madmom_approximately_double"
        elif 0.44 <= tempo_ratio <= 0.57:
            comparison["rateRelation"] = "madmom_approximately_half"
        else:
            comparison["rateRelation"] = "different"

    if (
        librosa_period is None
        or madmom_period is None
        or len(librosa_beats) < 3
        or len(madmom_beats) < 3
    ):
        return comparison

    if librosa_period <= madmom_period:
        fast_grid = librosa_beats
        slow_grid = madmom_beats
        fast_period = librosa_period
        slow_period = madmom_period
    else:
        fast_grid = madmom_beats
        slow_grid = librosa_beats
        fast_period = madmom_period
        slow_period = librosa_period

    period_ratio = slow_period / fast_period

    comparison["slowerToFasterPulsePeriodRatio"] = float(period_ratio)

    stride = int(round(period_ratio))

    if (
        stride < 1
        or stride > 4
        or abs(period_ratio - stride) / stride > 0.15
    ):
        return comparison

    comparison["expectedFastPulsesPerSlowPulse"] = stride

    overlap_start = max(
        float(fast_grid[0]),
        float(slow_grid[0]),
    )
    overlap_end = min(
        float(fast_grid[-1]),
        float(slow_grid[-1]),
    )

    if overlap_end <= overlap_start:
        return comparison

    slow_in_overlap = slow_grid[
        (slow_grid >= overlap_start)
        & (slow_grid <= overlap_end)
    ]

    if len(slow_in_overlap) < 3:
        return comparison

    best_fraction = None
    best_median_error = None
    tolerance_seconds = max(
        0.035,
        fast_period * 0.22,
    )

    for phase in range(stride):
        candidate_fast_grid = fast_grid[phase::stride]
        candidate_fast_grid = candidate_fast_grid[
            (candidate_fast_grid >= overlap_start)
            & (candidate_fast_grid <= overlap_end)
        ]

        if len(candidate_fast_grid) < 2:
            continue

        errors = []

        for slow_time in slow_in_overlap:
            right_index = int(
                np.searchsorted(candidate_fast_grid, slow_time)
            )

            possible_indices = []

            if right_index < len(candidate_fast_grid):
                possible_indices.append(right_index)

            if right_index > 0:
                possible_indices.append(right_index - 1)

            if not possible_indices:
                continue

            error = min(
                abs(candidate_fast_grid[index] - slow_time)
                for index in possible_indices
            )
            errors.append(float(error))

        if not errors:
            continue

        errors = np.asarray(errors, dtype=float)
        matched_fraction = float(
            np.mean(errors <= tolerance_seconds)
        )
        median_error = float(np.median(errors))

        if (
            best_fraction is None
            or matched_fraction > best_fraction
            or (
                matched_fraction == best_fraction
                and median_error < best_median_error
            )
        ):
            best_fraction = matched_fraction
            best_median_error = median_error

    if best_fraction is not None:
        comparison["matchedFraction"] = best_fraction
        comparison["medianAlignmentErrorMs"] = (
            best_median_error * 1000.0
        )

    return comparison


def make_even_subdivision_grid(beat_times, subdivisions_per_pulse):
    """Divide cada intervalo entre pulsos en partes iguales."""
    beat_times = finite_array(beat_times)
    subdivisions_per_pulse = int(subdivisions_per_pulse)

    if len(beat_times) < 2 or subdivisions_per_pulse < 2:
        return beat_times

    intervals = np.diff(beat_times)
    valid_intervals = intervals[
        np.isfinite(intervals)
        & (intervals > 0)
    ]

    if not valid_intervals.size:
        return beat_times

    final_interval = float(np.median(valid_intervals))
    extended_beats = np.append(
        beat_times,
        beat_times[-1] + final_interval,
    )

    interval_count = len(extended_beats) - 1
    grid = np.empty(
        interval_count * subdivisions_per_pulse + 1,
        dtype=float,
    )

    for subdivision in range(subdivisions_per_pulse):
        fraction = subdivision / subdivisions_per_pulse
        grid[subdivision:-1:subdivisions_per_pulse] = (
            extended_beats[:-1]
            + (extended_beats[1:] - extended_beats[:-1]) * fraction
        )

    grid[-1] = extended_beats[-1]
    return grid


def make_subdivision_grid(beat_times):
    """Intercala los pulsos y los puntos medios entre ellos."""
    return make_even_subdivision_grid(beat_times, 2)


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
    if len(beat_times) < 4:
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


def score_downbeat_period(probabilities, period):
    """Mide si el downbeat se repite en la misma posición de cada compás.

    Usa medianas por posición para que un downbeat aislado o una predicción
    atípica de Madmom no domine el resultado.
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

    bar_values = probabilities[:complete_bars * period].reshape(
        complete_bars,
        period,
    )
    position_profile = np.median(bar_values, axis=0)

    best_score = None

    for phase in range(period):
        downbeat_value = float(position_profile[phase])
        other_values = np.delete(position_profile, phase)

        if not len(other_values):
            continue

        score = downbeat_value - float(np.median(other_values))

        if np.isfinite(score) and (
            best_score is None or score > best_score
        ):
            best_score = float(score)

    return best_score


def score_accent_pattern(features, pattern):
    """Compara acentos repetidos por posición dentro del compás.

    La mediana por posición reduce el efecto de notas falsas y ataques
    aislados. También se exige que la figura métrica aparezca en al menos
    dos compases completos.
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

    if period < 2 or not np.all(np.isfinite(pattern)):
        return None

    finite_mask = np.isfinite(features)

    if np.count_nonzero(finite_mask) < period * 2:
        return None

    finite_mean = float(np.mean(features[finite_mask]))
    features = np.where(
        finite_mask,
        features,
        finite_mean,
    )

    complete_bars = len(features) // period

    if complete_bars < 2:
        return None

    bar_values = features[:complete_bars * period].reshape(
        complete_bars,
        period,
    )
    position_profile = np.median(bar_values, axis=0)
    profile_centered = position_profile - float(np.mean(position_profile))
    profile_scale = float(np.std(profile_centered))

    if not np.isfinite(profile_scale) or profile_scale < 1e-8:
        return None

    normalized_bars = bar_values - np.mean(
        bar_values,
        axis=1,
        keepdims=True,
    )
    bar_scales = np.std(normalized_bars, axis=1, keepdims=True)
    valid_bar_mask = bar_scales[:, 0] >= 1e-8
    normalized_bars = np.divide(
        normalized_bars,
        bar_scales,
        out=np.zeros_like(normalized_bars),
        where=bar_scales >= 1e-8,
    )

    best_score = None

    for phase in range(period):
        expected = pattern[(np.arange(period) - phase) % period]
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

        bar_scores = []

        if np.any(valid_bar_mask):
            expected_unit = expected_centered / expected_scale
            bar_scores = np.mean(
                normalized_bars[valid_bar_mask] * expected_unit,
                axis=1,
            ).tolist()

        # El perfil mediano identifica la jerarquía típica; la consistencia
        # por compás evita premiar una coincidencia que aparece una sola vez.
        consistency_score = (
            float(np.median(bar_scores))
            if bar_scores
            else profile_score
        )
        score = 0.75 * profile_score + 0.25 * consistency_score

        if (
            np.isfinite(score)
            and (best_score is None or score > best_score)
        ):
            best_score = score

    return best_score


def score_meter_hypothesis(
    beat_features,
    subdivision_features,
    downbeat_probabilities,
    downbeat_period,
    accent_pattern,
    use_subdivision_grid=False,
    hierarchy_features=None,
    hierarchy_pattern=None,
):
    """Combina jerarquía de acentos y evidencia de downbeat.

    hierarchy_features/pattern representan subdivisiones internas del pulso:
    cuartos dentro de una blanca o corcheas dentro de una negra con puntillo.
    """
    if use_subdivision_grid:
        accent_score = score_accent_pattern(
            subdivision_features,
            accent_pattern,
        )
    else:
        accent_score = score_accent_pattern(
            beat_features,
            accent_pattern,
        )

    downbeat_score = score_downbeat_period(
        downbeat_probabilities,
        downbeat_period,
    )

    hierarchy_score = None

    if hierarchy_features is not None and hierarchy_pattern is not None:
        hierarchy_score = score_accent_pattern(
            hierarchy_features,
            hierarchy_pattern,
        )

    components = []

    if accent_score is not None and np.isfinite(accent_score):
        components.append((0.45, accent_score))

    if hierarchy_score is not None and np.isfinite(hierarchy_score):
        components.append((0.30, hierarchy_score))

    if downbeat_score is not None and np.isfinite(downbeat_score):
        components.append((0.25, downbeat_score))

    if not components:
        return None

    total_weight = sum(
        weight
        for weight, _ in components
    )
    score = sum(
        weight * value
        for weight, value in components
    ) / total_weight

    return float(score) if np.isfinite(score) else None

METER_PULSE_VARIANTS = {
    (2, 4): [
        {
            "beatsPerBar": 2,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.18],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [1.0, 0.08, 0.45, 0.08],
        },
        {
            "beatsPerBar": 4,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.10, 0.45, 0.10],
            "subdivisionsPerPulse": 1,
        },
    ],
    (3, 4): [
        {
            "beatsPerBar": 3,
            "beatUnit": "quarter",
            "beatUnitQuarterLength": 1.0,
            "accentPattern": [1.0, 0.18, 0.28],
            "subdivisionsPerPulse": 2,
            "subdivisionAccentPattern": [
                1.0, 0.08, 0.28, 0.08, 0.18, 0.08,
            ],
        },
        {
            "beatsPerBar": 6,
            "beatUnit": "eighth",
            "beatUnitQuarterLength": 0.5,
            "accentPattern": [1.0, 0.08, 0.28, 0.08, 0.18, 0.08],
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
}

def estimate_meter_for_grid(
    audio_path,
    beat_times,
    onset_envelope,
    sr,
    bar_processor,
):
    """
    Evalúa cada compás con varias unidades de pulso posibles.

    beatsPerBar indica cuántos pulsos de la cuadrícula forman el compás.
    beatUnit indica la figura musical asignada a cada pulso.
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

            score = score_meter_hypothesis(
                beat_features=beat_features,
                subdivision_features=beat_features,
                downbeat_probabilities=downbeat_probabilities,
                downbeat_period=pulses_per_bar,
                accent_pattern=variant["accentPattern"],
                hierarchy_features=hierarchy_features,
                hierarchy_pattern=hierarchy_pattern,
            )

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


def read_midi_note_starts(midi_path):
    """Lee inicios de notas del MIDI si pretty_midi está disponible."""
    if not midi_path or not os.path.isfile(midi_path):
        return None

    try:
        import pretty_midi
    except ImportError:
        print(
            "AVISO: pretty_midi no está instalado; "
            "se omite el diagnóstico de alineación MIDI.",
            file=sys.stderr,
        )
        return None

    try:
        midi_file = pretty_midi.PrettyMIDI(midi_path)
    except Exception as exc:
        print(
            f"AVISO: no se pudo leer el MIDI: {exc}",
            file=sys.stderr,
        )
        return None

    starts = [
        float(note.start)
        for instrument in midi_file.instruments
        for note in instrument.notes
        if np.isfinite(note.start)
    ]

    return finite_array(starts)


def evaluate_midi_alignment(note_starts, beat_times):
    """
    Diagnostica la cercanía entre ataques MIDI y pulsos.

    Este dato no decide el compás ni elimina notas: los ataques fuera del
    pulso también pueden ser notas musicales válidas.
    """
    if note_starts is None:
        return {
            "noteCount": None,
            "checkedNoteCount": None,
            "fractionNearPulse": None,
            "medianDistanceAsBeatFraction": None,
        }

    beat_times = finite_array(beat_times)
    note_starts = finite_array(note_starts)
    period = grid_period(beat_times)

    if len(beat_times) < 2 or period is None:
        return {
            "noteCount": int(len(note_starts)),
            "checkedNoteCount": 0,
            "fractionNearPulse": None,
            "medianDistanceAsBeatFraction": None,
        }

    tolerance = period * 0.20
    normalized_distances = []

    for note_start in note_starts:
        if (
            note_start < beat_times[0]
            or note_start > beat_times[-1]
        ):
            continue

        right_index = int(
            np.searchsorted(beat_times, note_start)
        )
        possible_indices = []

        if right_index < len(beat_times):
            possible_indices.append(right_index)

        if right_index > 0:
            possible_indices.append(right_index - 1)

        if not possible_indices:
            continue

        distance = min(
            abs(beat_times[index] - note_start)
            for index in possible_indices
        )
        normalized_distances.append(distance / period)

    if not normalized_distances:
        return {
            "noteCount": int(len(note_starts)),
            "checkedNoteCount": 0,
            "fractionNearPulse": None,
            "medianDistanceAsBeatFraction": None,
        }

    normalized_distances = np.asarray(
        normalized_distances,
        dtype=float,
    )

    return {
        "noteCount": int(len(note_starts)),
        "checkedNoteCount": int(len(normalized_distances)),
        "fractionNearPulse": float(
            np.mean(normalized_distances <= 0.20)
        ),
        "medianDistanceAsBeatFraction": float(
            np.median(normalized_distances)
        ),
    }


MIN_TEMPO_HYPOTHESIS_MARGIN = 0.04
MIN_JOINT_METER_MARGIN = 0.04


def autocorrelation_score_for_tempo(
    onset_envelope,
    sr,
    tempo_bpm,
):
    """
    Puntúa la repetición de ataques de audio al periodo de un tempo.
    Regresa un valor entre 0 y 1; no interpreta por sí solo el compás.
    """
    if (
        tempo_bpm is None
        or not np.isfinite(tempo_bpm)
        or tempo_bpm <= 0
    ):
        return None

    values = finite_array(onset_envelope)

    if len(values) < 32:
        return None

    values = values - float(np.mean(values))
    nominal_lag = (
        60.0 * sr
        / (float(tempo_bpm) * HOP_LENGTH)
    )

    if nominal_lag < 3 or nominal_lag >= len(values) / 3:
        return None

    center_lag = int(round(nominal_lag))
    radius = max(1, int(round(center_lag * 0.04)))
    scores = []

    for lag in range(
        max(2, center_lag - radius),
        min(len(values) // 3, center_lag + radius) + 1,
    ):
        first = values[:-lag]
        second = values[lag:]

        denominator = (
            float(np.linalg.norm(first))
            * float(np.linalg.norm(second))
        )

        if denominator <= 1e-12:
            continue

        correlation = float(
            np.dot(first, second) / denominator
        )

        if np.isfinite(correlation):
            scores.append(
                float(np.clip(correlation, 0.0, 1.0))
            )

    return max(scores) if scores else None


def make_grids_for_candidate_tempo(
    source_grid,
    target_tempo,
):
    """
    Adapta una cuadrícula existente al tempo candidato:
    conserva los pulsos, toma uno de cada dos, o intercala medios pulsos.
    """
    source_grid = finite_array(source_grid)
    source_tempo = grid_tempo(source_grid)

    if (
        source_tempo is None
        or target_tempo is None
        or target_tempo <= 0
        or len(source_grid) < 8
    ):
        return []

    ratio = source_tempo / float(target_tempo)
    variants = []

    if 0.88 <= ratio <= 1.12:
        variants.append(source_grid)

    elif 1.75 <= ratio <= 2.25:
        # El candidato es aproximadamente la mitad del pulso fuente.
        variants.append(source_grid[::2])
        variants.append(source_grid[1::2])

    elif 0.44 <= ratio <= 0.57:
        # El candidato es aproximadamente el doble del pulso fuente.
        variants.append(make_subdivision_grid(source_grid))

    return [
        finite_array(grid)
        for grid in variants
        if len(grid) >= 8
    ]


def midi_alignment_score(note_starts, beat_times):
    """
    Convierte el diagnóstico de alineación MIDI en una puntuación 0..1.
    Es solo una evidencia más: notas fuera del pulso pueden ser correctas.
    """
    alignment = evaluate_midi_alignment(
        note_starts,
        beat_times,
    )

    fraction = alignment.get("fractionNearPulse")
    median_distance = alignment.get(
        "medianDistanceAsBeatFraction"
    )

    if fraction is None or median_distance is None:
        return None

    phase_score = 1.0 - min(
        1.0,
        2.0 * float(median_distance),
    )

    score = (
        0.55 * float(fraction)
        + 0.45 * phase_score
    )

    return float(np.clip(score, 0.0, 1.0))


def combine_available_scores(
    audio_score,
    midi_score,
    meter_score,
):
    """
    Combina evidencia de audio y compás.

    midi_score se conserva como diagnóstico, pero no decide el tempo:
    una cuadrícula con más pulsos puede acercar más notas por densidad.
    """
    components = []

    if audio_score is not None and np.isfinite(audio_score):
        components.append((0.50, float(audio_score)))

    if meter_score is not None and np.isfinite(meter_score):
        # Los puntajes métricos están aproximadamente en [-1, 1].
        normalized_meter = float(
            np.clip((float(meter_score) + 1.0) / 2.0, 0.0, 1.0)
        )
        components.append((0.50, normalized_meter))

    if not components:
        return None

    total_weight = sum(weight for weight, _ in components)

    if total_weight <= 0:
        return None

    return float(
        sum(weight * score for weight, score in components)
        / total_weight
    )

def canonical_tempo_from_grid_hypothesis(
    grid_pulse_bpm,
    meter_candidate,
):
    """
    Convierte la velocidad de la cuadrícula a BPM notados.

    Para 6/8 usa como unidad convencional la negra con puntillo.
    Para los compases /4 usa la negra.
    """
    if (
        grid_pulse_bpm is None
        or not np.isfinite(grid_pulse_bpm)
        or grid_pulse_bpm <= 0
    ):
        return None

    numerator = int(meter_candidate["numerator"])
    denominator = int(meter_candidate["denominator"])
    pulses_per_bar = int(meter_candidate["beatsPerBar"])

    if pulses_per_bar <= 0:
        return None

    bar_quarter_length = (
        numerator * 4.0 / denominator
    )

    grid_unit_quarter_length = (
        bar_quarter_length / pulses_per_bar
    )

    declared_unit_length = float(
        meter_candidate["beatUnitQuarterLength"]
    )

    if not np.isclose(
        grid_unit_quarter_length,
        declared_unit_length,
        rtol=0.05,
        atol=0.01,
    ):
        return None

    if (numerator, denominator) == (6, 8):
        tempo_unit = "dotted_quarter"
        tempo_unit_quarter_length = 1.5
    else:
        tempo_unit = "quarter"
        tempo_unit_quarter_length = 1.0

    canonical_beats_per_bar = (
        bar_quarter_length / tempo_unit_quarter_length
    )

    bar_duration_seconds = (
        pulses_per_bar * 60.0 / float(grid_pulse_bpm)
    )

    if bar_duration_seconds <= 0:
        return None

    canonical_tempo_bpm = (
        canonical_beats_per_bar
        * 60.0
        / bar_duration_seconds
    )

    if (
        not np.isfinite(canonical_tempo_bpm)
        or canonical_tempo_bpm <= 0
    ):
        return None

    return {
        "tempoBpm": float(canonical_tempo_bpm),
        "tempoUnit": tempo_unit,
        "gridPulseBpm": float(grid_pulse_bpm),
        "gridPulseUnit": meter_candidate["beatUnit"],
        "gridPulsesPerBar": pulses_per_bar,
        "barDurationSeconds": float(bar_duration_seconds),
    }

def select_joint_tempo_and_meter(
    tempo_candidates,
    source_grids,
    note_starts,
    onset_envelope,
    audio_path,
    sr,
    bar_processor,
):
    """
    Evalúa tempo notado, unidad del pulso y compás.

    La alineación MIDI queda como dato de diagnóstico. No suma puntos
    para elegir el tempo porque las cuadrículas más densas pueden acercar
    artificialmente más notas.
    """
    pair_results = {}

    for target_tempo in tempo_candidates:
        if (
            target_tempo is None
            or not np.isfinite(target_tempo)
            or target_tempo <= 0
        ):
            continue

        for source_name, source_grid in source_grids.items():
            candidate_grids = make_grids_for_candidate_tempo(
                source_grid,
                target_tempo,
            )

            for candidate_grid in candidate_grids:
                grid_pulse_bpm = grid_tempo(candidate_grid)

                if (
                    grid_pulse_bpm is None
                    or grid_pulse_bpm <= 0
                ):
                    continue

                audio_score = autocorrelation_score_for_tempo(
                    onset_envelope,
                    sr,
                    grid_pulse_bpm,
                )

                meter_result = estimate_meter_for_grid(
                    audio_path=audio_path,
                    beat_times=candidate_grid,
                    onset_envelope=onset_envelope,
                    sr=sr,
                    bar_processor=bar_processor,
                )

                pulse_hypotheses = meter_result.get(
                    "pulseHypotheses",
                    [],
                )

                if not pulse_hypotheses:
                    continue

                midi_score = midi_alignment_score(
                    note_starts,
                    candidate_grid,
                )

                for meter_candidate in pulse_hypotheses:
                    tempo_interpretation = (
                        canonical_tempo_from_grid_hypothesis(
                            grid_pulse_bpm,
                            meter_candidate,
                        )
                    )

                    if tempo_interpretation is None:
                        continue

                    canonical_tempo = round(
                        tempo_interpretation["tempoBpm"],
                        1,
                    )
                    tempo_unit = tempo_interpretation["tempoUnit"]

                    total_score = combine_available_scores(
                        audio_score,
                        midi_score,
                        meter_candidate["score"],
                    )

                    if total_score is None:
                        continue

                    numerator = int(
                        meter_candidate["numerator"]
                    )
                    denominator = int(
                        meter_candidate["denominator"]
                    )

                    hypothesis = {
                        "tempoBpm": canonical_tempo,
                        "tempoUnit": tempo_unit,
                        "gridPulseBpm": round(
                            grid_pulse_bpm,
                            1,
                        ),
                        "gridPulseUnit":
                            meter_candidate["beatUnit"],
                        "gridPulsesPerBar":
                            int(meter_candidate["beatsPerBar"]),
                        "barDurationSeconds": round(
                            tempo_interpretation[
                                "barDurationSeconds"
                            ],
                            4,
                        ),
                        "numerator": numerator,
                        "denominator": denominator,
                        "audioScore": audio_score,
                        "midiScore": midi_score,
                        "meterScore": float(
                            meter_candidate["score"]
                        ),
                        "totalScore": float(total_score),
                        "gridSource": str(source_name),
                        "isSelected": False,
                    }

                    # Cuadrículas de fuentes distintas pueden producir
                    # prácticamente el mismo BPM notado y compás.
                    key = None

                    for existing_key in pair_results:
                        same_signature = (
                            existing_key[1] == tempo_unit
                            and existing_key[2] == numerator
                            and existing_key[3] == denominator
                        )

                        tempo_difference = abs(
                            existing_key[0] - canonical_tempo
                        ) / max(existing_key[0], 1e-9)

                        if (
                            same_signature
                            and tempo_difference <= 0.02
                        ):
                            key = existing_key
                            break

                    if key is None:
                        key = (
                            canonical_tempo,
                            tempo_unit,
                            numerator,
                            denominator,
                        )

                    previous = pair_results.get(key)

                    if (
                        previous is None
                        or hypothesis["totalScore"]
                        > previous["hypothesis"]["totalScore"]
                    ):
                        pair_results[key] = {
                            "hypothesis": hypothesis,
                            "meterCandidates":
                                meter_result["candidates"],
                        }

    hypotheses = [
        value["hypothesis"]
        for value in pair_results.values()
    ]

    hypotheses.sort(
        key=lambda item: item["totalScore"],
        reverse=True,
    )

    empty_result = {
        "selectedTempoBpm": None,
        "selectedTempoUnit": None,
        "selectedPulseBpm": None,
        "selectedPulseUnit": None,
        "selectedGridPulsesPerBar": None,
        "tempoConfidence": None,
        "tempoDecisionReason": "no_valid_tempo_hypotheses",
        "selectedMeter": None,
        "meterConfidence": None,
        "meterDecisionReason": "no_valid_meter_hypotheses",
        "tempoHypotheses": [],
        "selectedMeterCandidates": [],
    }

    if not hypotheses:
        return empty_result

    best_score_by_tempo = {}

    for item in hypotheses:
        tempo_key = (
            item["tempoBpm"],
            item["tempoUnit"],
        )

        best_score_by_tempo[tempo_key] = max(
            best_score_by_tempo.get(tempo_key, -1.0),
            item["totalScore"],
        )

    ranked_tempos = sorted(
        best_score_by_tempo.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_tempo = None
    selected_tempo_unit = None

    if len(ranked_tempos) == 1:
        selected_tempo, selected_tempo_unit = ranked_tempos[0][0]
        tempo_margin = 1.0
        tempo_reason = "only_one_valid_tempo_hypothesis"
    else:
        tempo_margin = (
            ranked_tempos[0][1] - ranked_tempos[1][1]
        )

        if tempo_margin >= MIN_TEMPO_HYPOTHESIS_MARGIN:
            selected_tempo, selected_tempo_unit = (
                ranked_tempos[0][0]
            )
            tempo_reason = "joint_audio_meter_score"
        else:
            tempo_reason = "tempo_candidates_are_too_close"

    selected_meter = None
    meter_margin = None
    selected_meter_candidates = []
    selected_pulse_bpm = None
    selected_pulse_unit = None
    selected_grid_pulses_per_bar = None

    if selected_tempo is None:
        meter_reason = "tempo_is_ambiguous"
    else:
        matching_items = [
            item
            for item in hypotheses
            if (
                item["tempoBpm"] == selected_tempo
                and item["tempoUnit"] == selected_tempo_unit
            )
        ]

        scores_by_meter = {}

        for item in matching_items:
            meter_key = (
                item["numerator"],
                item["denominator"],
            )

            current = scores_by_meter.get(meter_key)

            if (
                current is None
                or item["totalScore"] > current["score"]
            ):
                result_for_pair = next(
                    (
                        value
                        for value in pair_results.values()
                        if value["hypothesis"] is item
                    ),
                    None,
                )

                scores_by_meter[meter_key] = {
                    "score": item["totalScore"],
                    "hypothesis": item,
                    "meterCandidates": (
                        result_for_pair["meterCandidates"]
                        if result_for_pair
                        else []
                    ),
                }

        ranked_meters = sorted(
            scores_by_meter.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        if len(ranked_meters) == 1:
            (numerator, denominator), best = ranked_meters[0]
            selected_meter = {
                "numerator": numerator,
                "denominator": denominator,
            }
            meter_margin = 1.0
            selected_meter_candidates = best["meterCandidates"]
            meter_reason = "one_meter_hypothesis_for_selected_tempo"

        elif ranked_meters:
            meter_margin = (
                ranked_meters[0][1]["score"]
                - ranked_meters[1][1]["score"]
            )

            if meter_margin >= MIN_JOINT_METER_MARGIN:
                (numerator, denominator), best = ranked_meters[0]
                selected_meter = {
                    "numerator": numerator,
                    "denominator": denominator,
                }
                selected_meter_candidates = best["meterCandidates"]
                meter_reason = "joint_tempo_meter_score"
            else:
                meter_reason = "meter_candidates_are_too_close"
        else:
            meter_reason = "no_meter_for_selected_tempo"

    matching_tempo_items = [
        item
        for item in hypotheses
        if (
            selected_tempo is not None
            and item["tempoBpm"] == selected_tempo
            and item["tempoUnit"] == selected_tempo_unit
            and (
                selected_meter is None
                or (
                    item["numerator"] == selected_meter["numerator"]
                    and item["denominator"] == selected_meter["denominator"]
                )
            )
        )
    ]

    if matching_tempo_items:
        best_tempo_item = max(
            matching_tempo_items,
            key=lambda item: item["totalScore"],
        )
        selected_pulse_bpm = best_tempo_item["gridPulseBpm"]
        selected_pulse_unit = best_tempo_item["gridPulseUnit"]
        selected_grid_pulses_per_bar = (
            best_tempo_item["gridPulsesPerBar"]
        )

    for item in hypotheses:
        item["isSelected"] = bool(
            selected_tempo is not None
            and selected_meter is not None
            and item["tempoBpm"] == selected_tempo
            and item["tempoUnit"] == selected_tempo_unit
            and item["numerator"]
                == selected_meter["numerator"]
            and item["denominator"]
                == selected_meter["denominator"]
        )

    return {
        "selectedTempoBpm": selected_tempo,
        "selectedTempoUnit": selected_tempo_unit,
        "selectedPulseBpm": selected_pulse_bpm,
        "selectedPulseUnit": selected_pulse_unit,
        "selectedGridPulsesPerBar":
            selected_grid_pulses_per_bar,
        "tempoConfidence": tempo_margin,
        "tempoDecisionReason": tempo_reason,
        "selectedMeter": selected_meter,
        "meterConfidence": meter_margin,
        "meterDecisionReason": meter_reason,
        "tempoHypotheses": hypotheses,
        "selectedMeterCandidates": selected_meter_candidates,
    }

def round_times(times):
    """Redondea tiempos para incluir las cuadrículas en el JSON."""
    return [
        round(float(value), 6)
        for value in finite_array(times)
    ]


def choose_consensus_meter(
    librosa_meter,
    madmom_meter,
    grid_comparison,
):
    """
    Confirma compás solo si las cuadrículas y ambas evaluaciones coinciden.
    """
    librosa_selected = librosa_meter["selected"]
    madmom_selected = madmom_meter["selected"]

    if not librosa_selected or not madmom_selected:
        return None, None, "one_or_both_grids_are_ambiguous"

    same_meter = (
        librosa_selected["numerator"]
        == madmom_selected["numerator"]
        and librosa_selected["denominator"]
        == madmom_selected["denominator"]
    )

    if not same_meter:
        return None, None, "grid_meter_predictions_disagree"

    rate_relation = grid_comparison["rateRelation"]
    matched_fraction = grid_comparison["matchedFraction"]

    if rate_relation != "similar":
        return None, None, "pulse_rates_disagree"

    if (
        matched_fraction is None
        or matched_fraction < 0.60
    ):
        return None, None, "pulse_positions_do_not_agree_enough"

    confidence = min(
        librosa_meter["margin"],
        madmom_meter["margin"],
    )

    return librosa_selected, float(confidence), "grid_consensus"


def main():
    if not AUDIO_PATH:
        raise RuntimeError(
            "Falta la variable de entorno AUDIO_INPUT."
        )

    (
        librosa_tempo,
        librosa_beats,
        y,
        sr,
    ) = estimate_librosa(AUDIO_PATH)

    madmom_beats = estimate_madmom_beats(AUDIO_PATH)
    madmom_tempo = grid_tempo(madmom_beats)

    tempo_candidates = build_tempo_candidates(
        librosa_tempo,
        madmom_tempo,
    )

    tempo_relation = describe_tempo_relation(
        librosa_tempo,
        madmom_tempo,
    )
    grid_comparison = compare_beat_grids(
        librosa_beats,
        madmom_beats,
    )

    onset_envelope = build_onset_envelope(y, sr)

    # Se reutiliza una instancia del procesador para ambas cuadrículas.
    try:
        bar_processor = RNNBarProcessor()
    except Exception as exc:
        bar_processor = None
        print(
            "AVISO: no se pudo iniciar RNNBarProcessor: "
            f"{exc}",
            file=sys.stderr,
        )

    if bar_processor is None:
        # Las funciones aceptan el procesador; este reemplazo permitirá que
        # el cálculo basado en acentos continúe sin downbeat de Madmom.
        class EmptyBarProcessor:
            def __call__(self, _value):
                raise RuntimeError(
                    "RNNBarProcessor no está disponible."
                )

        bar_processor = EmptyBarProcessor()

    librosa_meter = estimate_meter_for_grid(
        audio_path=AUDIO_PATH,
        beat_times=librosa_beats,
        onset_envelope=onset_envelope,
        sr=sr,
        bar_processor=bar_processor,
    )
    madmom_meter = estimate_meter_for_grid(
        audio_path=AUDIO_PATH,
        beat_times=madmom_beats,
        onset_envelope=onset_envelope,
        sr=sr,
        bar_processor=bar_processor,
    )

    note_starts = read_midi_note_starts(MIDI_PATH)

    tempo_selection = select_joint_tempo_and_meter(
        tempo_candidates=tempo_candidates,
        source_grids={
            "librosa": librosa_beats,
            "madmom": madmom_beats,
        },
        note_starts=note_starts,
        onset_envelope=onset_envelope,
        audio_path=AUDIO_PATH,
        sr=sr,
        bar_processor=bar_processor,
    )

    selected_tempo = tempo_selection["selectedTempoBpm"]
    selected_meter = tempo_selection["selectedMeter"]
    tempo_confidence = tempo_selection["tempoConfidence"]
    tempo_decision_reason = tempo_selection["tempoDecisionReason"]
    meter_confidence = tempo_selection["meterConfidence"]
    decision_reason = tempo_selection["meterDecisionReason"]
    selected_tempo_unit = tempo_selection["selectedTempoUnit"]
    selected_pulse_bpm = tempo_selection["selectedPulseBpm"]
    selected_pulse_unit = tempo_selection["selectedPulseUnit"]
    selected_grid_pulses_per_bar = (
        tempo_selection["selectedGridPulsesPerBar"]
    )

    # El compás puede confirmarse aunque el tempo siga ambiguo,
    # si ambas cuadrículas coinciden y sus análisis de compás concuerdan.
    if selected_meter is None:
        (
            consensus_meter,
            consensus_confidence,
            consensus_reason,
        ) = choose_consensus_meter(
            librosa_meter,
            madmom_meter,
            grid_comparison,
        )

        if consensus_meter is not None:
            selected_meter = consensus_meter
            meter_confidence = consensus_confidence
            decision_reason = (
                "meter_consensus_independent_of_tempo:"
                f"{consensus_reason}"
            )

            tempo_selection["selectedMeter"] = selected_meter
            tempo_selection["meterConfidence"] = meter_confidence
            tempo_selection["meterDecisionReason"] = decision_reason
            tempo_selection["selectedMeterCandidates"] = (
                librosa_meter["candidates"]
            )

    midi_alignment = {
        "librosaGrid": evaluate_midi_alignment(
            note_starts,
            librosa_beats,
        ),
        "madmomGrid": evaluate_midi_alignment(
            note_starts,
            madmom_beats,
        ),
        "note": (
            "Diagnóstico solamente; no se eliminaron ni corrigieron notas."
        ),
    }

    # Conserva la lista de candidatos de la cuadrícula principal
    # para mantener compatibilidad con el DTO actual de C#.
    primary_meter_candidates = librosa_meter["candidates"]

    if not primary_meter_candidates:
        primary_meter_candidates = madmom_meter["candidates"]

    if tempo_selection["selectedMeterCandidates"]:
        primary_meter_candidates = tempo_selection[
            "selectedMeterCandidates"
        ]

    result = {
        # Estimaciones originales de los rastreadores.
        "tempoBpm": librosa_tempo,
        "madmomTempoBpm": madmom_tempo,
        "tempoCandidatesBpm": tempo_candidates,

        # Tempo seleccionado y significado de la cuadrícula que lo respalda.
        "selectedTempoBpm": selected_tempo,
        "selectedTempoUnit": selected_tempo_unit,
        "selectedPulseBpm": selected_pulse_bpm,
        "selectedPulseUnit": selected_pulse_unit,
        "selectedGridPulsesPerBar": selected_grid_pulses_per_bar,
        "tempoConfidence": tempo_confidence,
        "tempoDecisionReason": tempo_decision_reason,
        "tempoHypotheses": tempo_selection["tempoHypotheses"],

        "beatCount": int(len(madmom_beats)),

        # El compás se representa en los campos que consume C#.
        "timeSignatureNumerator": (
            selected_meter["numerator"]
            if selected_meter
            else None
        ),
        "timeSignatureDenominator": (
            selected_meter["denominator"]
            if selected_meter
            else None
        ),
        "meterConfidence": meter_confidence,
        "meterCandidates": primary_meter_candidates,

        # Datos de diagnóstico adicionales.
        "tempoRelation": tempo_relation,
        "beatGridComparison": grid_comparison,
        "meterCandidatesByGrid": {
            "librosa": librosa_meter,
            "madmom": madmom_meter,
        },
        "meterDecisionReason": decision_reason,
        "librosaBeatTimes": round_times(librosa_beats),
        "madmomBeatTimes": round_times(madmom_beats),
        "midiGridAlignment": midi_alignment,
        }

    print("TEMPO LIBROSA:", librosa_tempo)
    print("TEMPO MADMOM:", madmom_tempo)
    print("CANDIDATOS TEMPO:", tempo_candidates)
    print("TEMPO SELECCIONADO:", selected_tempo)
    print("MOTIVO SELECCIÓN TEMPO:", tempo_decision_reason)
    print(
        "HIPÓTESIS TEMPO/COMPÁS:",
        json.dumps(
            tempo_selection["tempoHypotheses"][:8],
            allow_nan=False,
        ),
    )
    print(
        "RELACIÓN DE TEMPOS:",
        json.dumps(tempo_relation, allow_nan=False),
    )
    print(
        "COMPARACIÓN DE CUADRÍCULAS:",
        json.dumps(grid_comparison, allow_nan=False),
    )
    print(
        "COMPASES EN CUADRÍCULA LIBROSA:",
        json.dumps(librosa_meter, allow_nan=False),
    )
    print(
        "COMPASES EN CUADRÍCULA MADMOM:",
        json.dumps(madmom_meter, allow_nan=False),
    )
    print(
        "ALINEACIÓN MIDI:",
        json.dumps(midi_alignment, allow_nan=False),
    )

    if selected_meter is None:
        print(
            "COMPÁS CONFIRMADO: ambiguo;",
            decision_reason,
        )
    else:
        print(
            "COMPÁS CONFIRMADO:",
            f"{selected_meter['numerator']}/"
            f"{selected_meter['denominator']}",
        )

    print(
        "ANALYSIS_JSON:"
        + json.dumps(result, allow_nan=False)
    )


if __name__ == "__main__":
    main()
