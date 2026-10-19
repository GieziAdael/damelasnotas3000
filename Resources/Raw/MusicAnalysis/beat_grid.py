"""Creación y comparación de cuadrículas de pulsos."""
import numpy as np

from audio_features import finite_array

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
