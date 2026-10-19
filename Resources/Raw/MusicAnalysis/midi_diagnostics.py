"""Lectura de ataques y comparación diagnóstica con el MIDI."""
import os
import sys

import numpy as np

from audio_features import finite_array
from beat_grid import grid_period

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
