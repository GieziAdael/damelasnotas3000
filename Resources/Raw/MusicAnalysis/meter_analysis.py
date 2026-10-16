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


def make_subdivision_grid(beat_times):
    """Intercala los pulsos y los puntos medios entre ellos."""
    beat_times = finite_array(beat_times)

    if len(beat_times) < 2:
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

    grid = np.empty(
        2 * len(extended_beats) - 1,
        dtype=float,
    )
    grid[0::2] = extended_beats
    grid[1::2] = (
        extended_beats[:-1] + extended_beats[1:]
    ) / 2.0

    return grid


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

    if len(probabilities) == len(beat_times):
        return probabilities

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
    """Mide si la probabilidad de downbeat se repite con el periodo esperado."""
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

    best_score = None
    indices = np.arange(len(probabilities))

    for phase in range(period):
        downbeat_mask = (
            (indices - phase) % period
        ) == 0
        other_mask = ~downbeat_mask

        if (
            not np.any(downbeat_mask)
            or not np.any(other_mask)
        ):
            continue

        score = float(
            np.mean(probabilities[downbeat_mask])
            - np.mean(probabilities[other_mask])
        )

        if (
            np.isfinite(score)
            and (best_score is None or score > best_score)
        ):
            best_score = score

    return best_score


def score_accent_pattern(features, pattern):
    """Compara los acentos observados con un patrón de compás."""
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

    feature_centered = features - float(np.mean(features))
    feature_scale = float(np.std(feature_centered))

    if not np.isfinite(feature_scale) or feature_scale < 1e-8:
        return None

    best_score = None
    indices = np.arange(len(features))

    for phase in range(period):
        expected = pattern[
            (indices - phase) % period
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

        score = float(
            np.mean(feature_centered * expected_centered)
            / (feature_scale * expected_scale)
        )

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
):
    """Combina únicamente puntuaciones finitas."""
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

    components = []

    if accent_score is not None and np.isfinite(accent_score):
        components.append((0.60, accent_score))

    if downbeat_score is not None and np.isfinite(downbeat_score):
        components.append((0.40, downbeat_score))

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


def estimate_meter_for_grid(
    audio_path,
    beat_times,
    onset_envelope,
    sr,
    bar_processor,
):
    """Evalúa candidatos de compás sobre una cuadrícula de pulsos concreta."""
    beat_times = finite_array(beat_times)

    if len(beat_times) < 8:
        return {
            "selected": None,
            "margin": None,
            "candidates": [],
        }

    subdivision_grid = make_subdivision_grid(beat_times)

    beat_features = sample_onset_strength(
        onset_envelope,
        sr,
        beat_times,
    )
    subdivision_features = sample_onset_strength(
        onset_envelope,
        sr,
        subdivision_grid,
    )
    downbeat_probabilities = get_downbeat_probabilities(
        audio_path,
        beat_times,
        bar_processor,
    )

    hypotheses = [
        {
            "beatsPerBar": 2,
            "numerator": 2,
            "denominator": 4,
            "downbeatPeriod": 2,
            "pattern": [1.0, 0.15],
            "useSubdivisionGrid": False,
        },
        {
            "beatsPerBar": 3,
            "numerator": 3,
            "denominator": 4,
            "downbeatPeriod": 3,
            "pattern": [1.0, 0.15, 0.10],
            "useSubdivisionGrid": False,
        },
        {
            "beatsPerBar": 4,
            "numerator": 4,
            "denominator": 4,
            "downbeatPeriod": 4,
            "pattern": [1.0, 0.15, 0.60, 0.15],
            "useSubdivisionGrid": False,
        },
        {
            "beatsPerBar": 6,
            "numerator": 6,
            "denominator": 8,
            "downbeatPeriod": 3,
            "pattern": [
                1.0,
                0.10,
                0.15,
                0.60,
                0.10,
                0.15,
            ],
            "useSubdivisionGrid": True,
        },
    ]

    candidates = []

    for hypothesis in hypotheses:
        score = score_meter_hypothesis(
            beat_features=beat_features,
            subdivision_features=subdivision_features,
            downbeat_probabilities=downbeat_probabilities,
            downbeat_period=hypothesis["downbeatPeriod"],
            accent_pattern=hypothesis["pattern"],
            use_subdivision_grid=hypothesis[
                "useSubdivisionGrid"
            ],
        )

        if score is None or not np.isfinite(score):
            continue

        candidates.append(
            {
                "numerator": hypothesis["numerator"],
                "denominator": hypothesis["denominator"],
                "beatsPerBar": hypothesis["beatsPerBar"],
                "score": float(score),
            }
        )

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    if len(candidates) < 2:
        return {
            "selected": None,
            "margin": None,
            "candidates": candidates,
        }

    margin = candidates[0]["score"] - candidates[1]["score"]

    if not np.isfinite(margin):
        margin = None
    elif margin < MIN_METER_SCORE_MARGIN:
        margin = float(margin)
    else:
        margin = float(margin)

    selected = None

    if (
        margin is not None
        and margin >= MIN_METER_SCORE_MARGIN
    ):
        selected = {
            "numerator": candidates[0]["numerator"],
            "denominator": candidates[0]["denominator"],
        }

    return {
        "selected": selected,
        "margin": margin,
        "candidates": candidates,
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

    selected_meter, meter_confidence, decision_reason = (
        choose_consensus_meter(
            librosa_meter,
            madmom_meter,
            grid_comparison,
        )
    )

    note_starts = read_midi_note_starts(MIDI_PATH)

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

    result = {
        "tempoBpm": librosa_tempo,
        "madmomTempoBpm": madmom_tempo,
        "tempoCandidatesBpm": tempo_candidates,

        # Se conserva el conteo anterior de pulsos Madmom.
        "beatCount": int(len(madmom_beats)),

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

        # Diagnósticos adicionales. El C# actual puede ignorarlos.
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