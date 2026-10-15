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

# Umbral inicial. Conviene calibrarlo con grabaciones de compás conocido.
MIN_METER_SCORE_MARGIN = 0.12
PROBABILITY_EPSILON = 1e-6


def estimate_librosa_tempo(audio_path):
    """Calcula el tempo de Librosa y carga el audio para reutilizarlo."""
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"No se encontró el audio: {audio_path}"
        )

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    tempo_values, _ = librosa.beat.beat_track(
        y=y,
        sr=sr,
    )

    tempo_values = np.asarray(
        tempo_values,
        dtype=float,
    ).reshape(-1)

    if not tempo_values.size:
        return None, y, sr

    tempo = float(tempo_values[0])

    if not np.isfinite(tempo) or tempo <= 0:
        return None, y, sr

    return tempo, y, sr


def estimate_madmom_beats(audio_path):
    """Obtiene los tiempos de pulso usando Madmom."""
    activations = RNNBeatProcessor()(audio_path)

    tracker = DBNBeatTrackingProcessor(
        fps=100,
        min_bpm=50,
        max_bpm=220,
    )

    beat_times = np.asarray(
        tracker(activations),
        dtype=float,
    ).reshape(-1)

    return beat_times[np.isfinite(beat_times)]


def estimate_madmom_tempo(beat_times):
    """Calcula BPM a partir de los intervalos entre pulsos de Madmom."""
    beat_times = np.asarray(
        beat_times,
        dtype=float,
    ).reshape(-1)

    if len(beat_times) < 2:
        return None

    intervals = np.diff(beat_times)
    valid_intervals = intervals[
        np.isfinite(intervals)
        & (intervals >= 60.0 / 220.0)
        & (intervals <= 60.0 / 50.0)
    ]

    if not valid_intervals.size:
        return None

    tempo = 60.0 / float(np.median(valid_intervals))

    if not np.isfinite(tempo) or tempo <= 0:
        return None

    return tempo


def build_tempo_candidates(librosa_tempo, madmom_tempo):
    """Conserva los tempos estimados y sus alternativas a mitad o al doble."""
    raw_tempos = [
        value
        for value in (librosa_tempo, madmom_tempo)
        if value is not None and np.isfinite(value)
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

        is_duplicate = any(
            abs(rounded_value - existing) / existing <= 0.02
            for existing in candidates
        )

        if not is_duplicate:
            candidates.append(rounded_value)

    return candidates


def robust_normalize(values):
    """Normaliza valores y convierte entradas no finitas en valores neutrales."""
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


def make_eighth_note_grid(beat_times):
    """
    Crea una cuadrícula con los pulsos detectados y sus puntos intermedios.

    Los puntos intermedios permiten evaluar el agrupamiento de 6/8 en
    dos grupos de tres corcheas.
    """
    beat_times = np.asarray(
        beat_times,
        dtype=float,
    ).reshape(-1)

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


def sample_onset_strength(y, sr, grid_times):
    """Mide la energía de ataque del audio cerca de cada punto de la cuadrícula."""
    grid_times = np.asarray(
        grid_times,
        dtype=float,
    ).reshape(-1)

    if not len(grid_times):
        return np.asarray([], dtype=float)

    hop_length = 512

    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    if not len(onset_envelope):
        return np.zeros(
            len(grid_times),
            dtype=float,
        )

    radius = max(
        1,
        int(round(0.06 * sr / hop_length)),
    )

    strengths = np.zeros(
        len(grid_times),
        dtype=float,
    )

    for index, time_seconds in enumerate(grid_times):
        if not np.isfinite(time_seconds):
            continue

        frame = int(round(time_seconds * sr / hop_length))
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


def sample_midi_onsets(midi_path, grid_times):
    """
    Mide ataques del MIDI cerca de cada punto de la cuadrícula.

    pretty_midi es opcional. Si no está instalado o no se puede leer el MIDI,
    el análisis continúa usando los acentos del audio.
    """
    grid_times = np.asarray(
        grid_times,
        dtype=float,
    ).reshape(-1)

    if (
        not midi_path
        or not os.path.isfile(midi_path)
        or not len(grid_times)
    ):
        return None

    try:
        import pretty_midi
    except ImportError:
        print(
            "AVISO: pretty_midi no está instalado; "
            "el análisis de compás usará solo el audio.",
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

    note_starts = []
    note_weights = []

    for instrument in midi_file.instruments:
        for note in instrument.notes:
            if (
                np.isfinite(note.start)
                and np.isfinite(note.velocity)
            ):
                note_starts.append(float(note.start))
                note_weights.append(
                    max(0.0, float(note.velocity) / 127.0)
                )

    if not note_starts:
        return None

    note_starts = np.asarray(
        note_starts,
        dtype=float,
    )
    note_weights = np.asarray(
        note_weights,
        dtype=float,
    )

    if len(grid_times) >= 2:
        grid_intervals = np.diff(grid_times)
        valid_intervals = grid_intervals[
            np.isfinite(grid_intervals)
            & (grid_intervals > 0)
        ]

        typical_interval = (
            float(np.median(valid_intervals))
            if valid_intervals.size
            else 0.5
        )
    else:
        typical_interval = 0.5

    tolerance = max(
        0.025,
        min(0.12, typical_interval * 0.18),
    )

    strengths = np.zeros(
        len(grid_times),
        dtype=float,
    )

    for note_start, weight in zip(note_starts, note_weights):
        right_index = int(
            np.searchsorted(grid_times, note_start)
        )

        possible_indices = []

        if right_index < len(grid_times):
            possible_indices.append(right_index)

        if right_index > 0:
            possible_indices.append(right_index - 1)

        if not possible_indices:
            continue

        nearest_index = min(
            possible_indices,
            key=lambda index: abs(
                grid_times[index] - note_start
            ),
        )

        distance = abs(
            grid_times[nearest_index] - note_start
        )

        if distance <= tolerance:
            strengths[nearest_index] += weight

    return robust_normalize(strengths)


def combine_accent_features(audio_features, midi_features):
    """Combina acentos del audio y del MIDI cuando ambos aportan señal."""
    audio_features = robust_normalize(audio_features)

    if midi_features is None:
        return audio_features, False

    midi_features = robust_normalize(midi_features)

    audio_has_variation = (
        len(audio_features) > 0
        and float(np.std(audio_features)) > 1e-8
    )
    midi_has_variation = (
        len(midi_features) > 0
        and float(np.std(midi_features)) > 1e-8
    )

    if not midi_has_variation:
        return audio_features, False

    if not audio_has_variation:
        return midi_features, True

    combined = (
        0.75 * audio_features
        + 0.25 * midi_features
    )

    return robust_normalize(combined), True


def align_downbeat_probabilities(bar_activations, beat_times):
    """Alinea las probabilidades de downbeat y descarta datos inválidos."""
    activations = np.asarray(
        bar_activations,
        dtype=float,
    )
    beat_times = np.asarray(
        beat_times,
        dtype=float,
    ).reshape(-1)

    if (
        activations.ndim != 2
        or activations.shape[1] < 2
        or not len(beat_times)
    ):
        return None

    probabilities = activations[:, 1]

    if not np.any(np.isfinite(probabilities)):
        return None

    # Una probabilidad inválida se trata como neutral.
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


def score_downbeat_period(probabilities, period):
    """
    Evalúa si las probabilidades de downbeat se repiten cada cierto número
    de pulsos. En 6/8 se usa un downbeat cada tres pulsos de negra.
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
    """Compara los acentos observados contra un patrón métrico."""
    features = np.asarray(
        features,
        dtype=float,
    ).reshape(-1)
    pattern = np.asarray(
        pattern,
        dtype=float,
    ).reshape(-1)

    period = len(pattern)

    if period < 2:
        return None

    finite_mask = np.isfinite(features)

    if np.count_nonzero(finite_mask) < period * 2:
        return None

    if not np.all(np.isfinite(pattern)):
        return None

    # Los puntos inválidos se sustituyen por el promedio para que no
    # introduzcan un acento artificial.
    finite_mean = float(np.mean(features[finite_mask]))
    features = np.where(
        finite_mask,
        features,
        finite_mean,
    )

    best_score = None
    indices = np.arange(len(features))
    feature_centered = features - float(np.mean(features))
    feature_scale = float(np.std(feature_centered))

    if not np.isfinite(feature_scale) or feature_scale < 1e-8:
        return None

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
    beat_accent_features,
    eighth_accent_features,
    downbeat_probabilities,
    beats_per_bar,
    accent_pattern,
    use_eighth_grid=False,
):
    """Combina puntuaciones finitas de acentos y downbeats."""
    if use_eighth_grid:
        accent_score = score_accent_pattern(
            eighth_accent_features,
            accent_pattern,
        )
    else:
        accent_score = score_accent_pattern(
            beat_accent_features,
            accent_pattern,
        )

    downbeat_score = score_downbeat_period(
        downbeat_probabilities,
        beats_per_bar,
    )

    components = []

    if (
        accent_score is not None
        and np.isfinite(accent_score)
    ):
        components.append((0.60, accent_score))

    if (
        downbeat_score is not None
        and np.isfinite(downbeat_score)
    ):
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


def estimate_meter_candidates(
    audio_path,
    beat_times,
    y,
    sr,
    midi_path,
):
    """Compara 2/4, 3/4, 4/4 y 6/8 usando el audio y el MIDI disponible."""
    beat_times = np.asarray(
        beat_times,
        dtype=float,
    ).reshape(-1)
    beat_times = beat_times[np.isfinite(beat_times)]

    if len(beat_times) < 8:
        return [], None, None, None, False

    eighth_grid = make_eighth_note_grid(beat_times)

    audio_beat_features = sample_onset_strength(
        y,
        sr,
        beat_times,
    )
    audio_eighth_features = sample_onset_strength(
        y,
        sr,
        eighth_grid,
    )

    midi_beat_features = sample_midi_onsets(
        midi_path,
        beat_times,
    )
    midi_eighth_features = sample_midi_onsets(
        midi_path,
        eighth_grid,
    )

    beat_features, midi_used_beats = combine_accent_features(
        audio_beat_features,
        midi_beat_features,
    )
    eighth_features, midi_used_eighths = combine_accent_features(
        audio_eighth_features,
        midi_eighth_features,
    )

    midi_used = midi_used_beats or midi_used_eighths
    downbeat_probabilities = None

    try:
        bar_activations = RNNBarProcessor()(
            (audio_path, beat_times)
        )

        downbeat_probabilities = align_downbeat_probabilities(
            bar_activations,
            beat_times,
        )
    except Exception as exc:
        print(
            "AVISO: no se pudieron obtener probabilidades "
            f"de downbeat de Madmom: {exc}",
            file=sys.stderr,
        )

    hypotheses = [
        {
            "beatsPerBar": 2,
            "numerator": 2,
            "denominator": 4,
            "downbeatPeriod": 2,
            "pattern": [1.0, 0.15],
            "useEighthGrid": False,
        },
        {
            "beatsPerBar": 3,
            "numerator": 3,
            "denominator": 4,
            "downbeatPeriod": 3,
            "pattern": [1.0, 0.15, 0.10],
            "useEighthGrid": False,
        },
        {
            "beatsPerBar": 4,
            "numerator": 4,
            "denominator": 4,
            "downbeatPeriod": 4,
            "pattern": [1.0, 0.15, 0.60, 0.15],
            "useEighthGrid": False,
        },
        {
            # Seis corcheas agrupadas en dos grupos de tres.
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
            "useEighthGrid": True,
        },
    ]

    candidates = []

    for hypothesis in hypotheses:
        score = score_meter_hypothesis(
            beat_accent_features=beat_features,
            eighth_accent_features=eighth_features,
            downbeat_probabilities=downbeat_probabilities,
            beats_per_bar=hypothesis["downbeatPeriod"],
            accent_pattern=hypothesis["pattern"],
            use_eighth_grid=hypothesis["useEighthGrid"],
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
        return candidates, None, None, None, midi_used

    best = candidates[0]
    second = candidates[1]
    score_margin = best["score"] - second["score"]

    if (
        not np.isfinite(score_margin)
        or score_margin < MIN_METER_SCORE_MARGIN
    ):
        safe_margin = (
            float(score_margin)
            if np.isfinite(score_margin)
            else None
        )
        return (
            candidates,
            None,
            None,
            safe_margin,
            midi_used,
        )

    return (
        candidates,
        best["numerator"],
        best["denominator"],
        float(score_margin),
        midi_used,
    )


def main():
    if not AUDIO_PATH:
        raise RuntimeError(
            "Falta la variable de entorno AUDIO_INPUT."
        )

    librosa_tempo, y, sr = estimate_librosa_tempo(
        AUDIO_PATH
    )

    beat_times = estimate_madmom_beats(
        AUDIO_PATH
    )

    madmom_tempo = estimate_madmom_tempo(
        beat_times
    )

    tempo_candidates = build_tempo_candidates(
        librosa_tempo,
        madmom_tempo,
    )

    (
        meter_candidates,
        selected_numerator,
        selected_denominator,
        meter_confidence,
        midi_used,
    ) = estimate_meter_candidates(
        audio_path=AUDIO_PATH,
        beat_times=beat_times,
        y=y,
        sr=sr,
        midi_path=MIDI_PATH,
    )

    result = {
        # Se conserva Librosa como tempo principal.
        "tempoBpm": librosa_tempo,
        "madmomTempoBpm": madmom_tempo,
        "tempoCandidatesBpm": tempo_candidates,

        "beatCount": int(len(beat_times)),

        # Quedan null si la evidencia no es suficiente.
        "timeSignatureNumerator": selected_numerator,
        "timeSignatureDenominator": selected_denominator,
        "meterConfidence": meter_confidence,

        "meterCandidates": meter_candidates,
        "midiRhythmUsed": bool(midi_used),
    }

    print("TEMPO LIBROSA:", librosa_tempo)
    print("TEMPO MADMOM:", madmom_tempo)
    print("CANDIDATOS TEMPO:", tempo_candidates)
    print("RITMO MIDI USADO:", midi_used)
    print("CANDIDATOS COMPÁS:", meter_candidates)

    if selected_numerator is None:
        print("COMPÁS CONFIRMADO: ambiguo")
    else:
        print(
            "COMPÁS CONFIRMADO:",
            f"{selected_numerator}/{selected_denominator}",
        )

    # allow_nan=False impide emitir JSON que .NET no pueda deserializar.
    print(
        "ANALYSIS_JSON:"
        + json.dumps(result, allow_nan=False)
    )


if __name__ == "__main__":
    main()