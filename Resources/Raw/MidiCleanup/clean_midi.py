import json
import math
import os
import sys

import librosa
import numpy as np
import pretty_midi


AUDIO_PATH = os.environ.get("AUDIO_INPUT")
MIDI_PATH = os.environ.get("MIDI_INPUT")
CLEANED_MIDI_PATH = os.environ.get("CLEANED_MIDI_OUTPUT")

SAMPLE_RATE = 22050
HOP_LENGTH = 256
PITCH_N_FFT = 4096
ONSET_N_FFT = 2048
HARMONICS_TO_CHECK = 8
EPSILON = 1e-10

# La clasificación "supported" y "review" solo se usa en el reporte.
SUPPORTED_MIN_DB = 3.0
REVIEW_MAX_DB = -6.0

# Eliminación automática muy conservadora:
# se exige que ambas medidas sean muy bajas y que la nota dure al menos
# 100 ms para no borrar notas breves por falta de resolución temporal.
AUTO_REMOVE_MAX_PITCH_DB = -15.0
AUTO_REMOVE_MAX_ONSET_DB = -12.0
AUTO_REMOVE_MIN_DURATION_SECONDS = 0.10

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def finite_or_none(value, digits=3):
    if value is None:
        return None

    numeric = float(value)

    if not math.isfinite(numeric):
        return None

    return round(numeric, digits)


def midi_pitch_name(pitch):
    pitch_class = int(pitch) % 12
    octave = int(pitch) // 12 - 1
    return f"{NOTE_NAMES[pitch_class]}{octave}"


def frames_between(
    start_seconds,
    end_seconds,
    sample_rate,
    hop_length,
    frame_count
):
    if end_seconds <= start_seconds:
        return np.empty(0, dtype=int)

    first = int(math.floor(
        start_seconds * sample_rate / hop_length
    ))

    last = int(math.ceil(
        end_seconds * sample_rate / hop_length
    )) + 1

    first = max(0, first)
    last = min(frame_count, last)

    if last <= first:
        return np.empty(0, dtype=int)

    return np.arange(first, last, dtype=int)


def build_pitch_salience(magnitude, sample_rate, n_fft):
    """
    Estima la energía de cada nota MIDI sumando sus armónicos.
    """
    frame_count = magnitude.shape[1]
    bin_width = float(sample_rate) / float(n_fft)

    salience = np.zeros(
        (128, frame_count),
        dtype=np.float32
    )

    nyquist = sample_rate / 2.0

    for pitch in range(128):
        fundamental_hz = 440.0 * (
            2.0 ** ((pitch - 69) / 12.0)
        )

        weighted_energy = np.zeros(
            frame_count,
            dtype=np.float32
        )

        total_weight = 0.0

        for harmonic in range(1, HARMONICS_TO_CHECK + 1):
            frequency = fundamental_hz * harmonic

            if frequency >= nyquist:
                break

            center_bin = int(round(frequency / bin_width))
            radius = max(
                1,
                int(round((frequency * 0.008) / bin_width))
            )

            low_bin = max(0, center_bin - radius)
            high_bin = min(
                magnitude.shape[0] - 1,
                center_bin + radius
            )

            if high_bin < low_bin:
                continue

            peak = np.max(
                magnitude[low_bin:high_bin + 1, :],
                axis=0
            )

            weight = 1.0 / float(harmonic)
            weighted_energy += peak * weight
            total_weight += weight

        if total_weight > 0:
            salience[pitch, :] = (
                weighted_energy / total_weight
            )

    return salience


def median_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(np.median(values))


def ratio_to_db(numerator, denominator):
    if numerator is None or denominator is None:
        return None

    numerator = float(numerator)
    denominator = float(denominator)

    if not math.isfinite(numerator):
        return None

    if not math.isfinite(denominator):
        return None

    if numerator <= EPSILON and denominator <= EPSILON:
        return None

    if numerator <= EPSILON:
        return -60.0

    if denominator <= EPSILON:
        return 60.0

    value = 20.0 * math.log10(
        numerator / denominator
    )

    return max(-60.0, min(60.0, value))


def calculate_pitch_contrast(
    pitch_scores,
    pitch,
    event_frames
):
    if event_frames.size == 0:
        return None

    target_level = median_or_none(
        pitch_scores[pitch, event_frames]
    )

    if target_level is None:
        return None

    neighbor_levels = []

    for neighbor_pitch in (
        pitch - 2,
        pitch - 1,
        pitch + 1,
        pitch + 2
    ):
        if 0 <= neighbor_pitch <= 127:
            level = median_or_none(
                pitch_scores[neighbor_pitch, event_frames]
            )

            if level is not None:
                neighbor_levels.append(level)

    if not neighbor_levels:
        return None

    neighbor_level = float(np.median(neighbor_levels))

    return ratio_to_db(
        target_level,
        neighbor_level
    )


def calculate_onset_rise(
    onset_scores,
    pitch,
    start_seconds,
    sample_rate,
    hop_length
):
    """
    Compara la energía antes y después del inicio MIDI.
    Prueba pequeños desplazamientos para tolerar diferencias de unos
    milisegundos entre el ataque del audio y el inicio del MIDI.
    """
    frame_count = onset_scores.shape[1]
    rises = []

    for shift in (-0.04, -0.02, 0.0, 0.02, 0.04):
        shifted_start = start_seconds + shift

        before_frames = frames_between(
            shifted_start - 0.14,
            shifted_start - 0.06,
            sample_rate,
            hop_length,
            frame_count
        )

        after_frames = frames_between(
            shifted_start + 0.02,
            shifted_start + 0.10,
            sample_rate,
            hop_length,
            frame_count
        )

        if before_frames.size < 2 or after_frames.size < 2:
            continue

        before_level = median_or_none(
            onset_scores[pitch, before_frames]
        )

        after_level = median_or_none(
            onset_scores[pitch, after_frames]
        )

        rise = ratio_to_db(after_level, before_level)

        if rise is not None:
            rises.append(rise)

    if not rises:
        return None

    # Se conserva la mejor evidencia encontrada en la ventana cercana.
    return max(rises)


def make_note_data(
    event_index,
    instrument_index,
    instrument,
    note
):
    instrument_name = instrument.name

    if not instrument_name:
        instrument_name = f"Instrumento {instrument_index + 1}"

    return {
        "eventIndex": int(event_index),
        "instrumentIndex": int(instrument_index),
        "instrumentName": str(instrument_name),
        "program": int(instrument.program),
        "isDrum": bool(instrument.is_drum),
        "pitch": int(note.pitch),
        "pitchName": midi_pitch_name(note.pitch),
        "startSeconds": finite_or_none(note.start, 6),
        "durationSeconds": finite_or_none(
            max(0.0, note.end - note.start),
            6
        ),
        "velocity": int(note.velocity)
    }


def assess_note(
    note_data,
    pitch_scores,
    onset_scores,
    sample_rate,
    hop_length,
    audio_duration
):
    pitch = int(note_data["pitch"])
    start = float(note_data["startSeconds"])
    duration = float(note_data["durationSeconds"])
    end = start + duration

    if note_data["isDrum"]:
        return {
            "note": note_data,
            "pitchContrastDb": None,
            "onsetRiseDb": None,
            "status": "inconclusive",
            "reason": (
                "Es percusión; este análisis tonal no está diseñado "
                "para evaluarla. Se conserva."
            )
        }

    if start >= audio_duration or end <= 0:
        return {
            "note": note_data,
            "pitchContrastDb": None,
            "onsetRiseDb": None,
            "status": "inconclusive",
            "reason": (
                "El evento queda fuera del audio disponible. "
                "Se conserva."
            )
        }

    edge = min(0.03, duration * 0.15)
    interior_start = start + edge
    interior_end = end - edge

    if interior_end <= interior_start:
        interior_start = start
        interior_end = end

    event_frames = frames_between(
        interior_start,
        interior_end,
        sample_rate,
        hop_length,
        pitch_scores.shape[1]
    )

    pitch_contrast_db = calculate_pitch_contrast(
        pitch_scores,
        pitch,
        event_frames
    )

    onset_rise_db = calculate_onset_rise(
        onset_scores,
        pitch,
        start,
        sample_rate,
        hop_length
    )

    if (
        duration >= AUTO_REMOVE_MIN_DURATION_SECONDS
        and pitch_contrast_db is not None
        and onset_rise_db is not None
        and pitch_contrast_db <= AUTO_REMOVE_MAX_PITCH_DB
        and onset_rise_db <= AUTO_REMOVE_MAX_ONSET_DB
    ):
        status = "removed"
        reason = (
            "Se eliminó de la copia porque la altura y el ataque "
            "tienen evidencia extremadamente baja en el audio. "
            "El MIDI original permanece intacto."
        )
    elif (
        pitch_contrast_db is not None
        and onset_rise_db is not None
        and pitch_contrast_db >= SUPPORTED_MIN_DB
        and onset_rise_db >= SUPPORTED_MIN_DB
    ):
        status = "supported"
        reason = (
            "La altura sobresale de las alturas vecinas y se observa "
            "un aumento de energía cerca del ataque. Se conserva."
        )
    elif (
        pitch_contrast_db is not None
        and onset_rise_db is not None
        and pitch_contrast_db <= REVIEW_MAX_DB
        and onset_rise_db <= REVIEW_MAX_DB
    ):
        status = "review"
        reason = (
            "Tiene poca evidencia de altura y ataque, pero no alcanza "
            "el umbral estricto para eliminarse automáticamente. "
            "Se conserva para revisión."
        )
    else:
        status = "inconclusive"
        reason = (
            "La evidencia es insuficiente o mixta. Se conserva."
        )

    return {
        "note": note_data,
        "pitchContrastDb": finite_or_none(pitch_contrast_db),
        "onsetRiseDb": finite_or_none(onset_rise_db),
        "status": status,
        "reason": reason
    }


def write_result(
    midi,
    output_path,
    assessments
):
    output_directory = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(output_directory, exist_ok=True)
    midi.write(output_path)

    result = {
        "totalNotes": len(assessments),
        "supportedCount": sum(
            1 for item in assessments
            if item["status"] == "supported"
        ),
        "reviewCount": sum(
            1 for item in assessments
            if item["status"] == "review"
        ),
        "inconclusiveCount": sum(
            1 for item in assessments
            if item["status"] == "inconclusive"
        ),
        "removedCount": sum(
            1 for item in assessments
            if item["status"] == "removed"
        ),
        "cleanedMidiPath": output_path,
        "notes": assessments
    }

    print(
        "CLEANUP_JSON:" +
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":")
        ),
        flush=True
    )


def main():
    if not AUDIO_PATH or not os.path.isfile(AUDIO_PATH):
        raise FileNotFoundError(
            f"No se encontró el audio: {AUDIO_PATH}"
        )

    if not MIDI_PATH or not os.path.isfile(MIDI_PATH):
        raise FileNotFoundError(
            f"No se encontró el MIDI: {MIDI_PATH}"
        )

    if not CLEANED_MIDI_PATH:
        raise ValueError(
            "No se indicó la ruta para el MIDI limpio."
        )

    source_path = os.path.normcase(os.path.abspath(MIDI_PATH))
    output_path = os.path.normcase(
        os.path.abspath(CLEANED_MIDI_PATH)
    )

    if source_path == output_path:
        raise ValueError(
            "La salida no puede sobrescribir el MIDI original."
        )

    audio, sample_rate = librosa.load(
        AUDIO_PATH,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("El archivo de audio está vacío.")

    audio_duration = float(audio.size) / float(sample_rate)
    midi = pretty_midi.PrettyMIDI(MIDI_PATH)

    raw_notes = []

    for instrument_index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            if note.end <= note.start:
                continue

            raw_notes.append(
                (
                    float(note.start),
                    int(instrument_index),
                    instrument,
                    note
                )
            )

    raw_notes.sort(
        key=lambda item: (
            item[0],
            item[1],
            int(item[3].pitch)
        )
    )

    if not raw_notes:
        write_result(
            midi,
            CLEANED_MIDI_PATH,
            []
        )
        return

    pitch_spectrum = librosa.stft(
        audio,
        n_fft=PITCH_N_FFT,
        hop_length=HOP_LENGTH,
        center=True
    )

    pitch_scores = build_pitch_salience(
        np.abs(pitch_spectrum).astype(np.float32),
        sample_rate,
        PITCH_N_FFT
    )

    del pitch_spectrum

    onset_spectrum = librosa.stft(
        audio,
        n_fft=ONSET_N_FFT,
        hop_length=HOP_LENGTH,
        center=True
    )

    onset_scores = build_pitch_salience(
        np.abs(onset_spectrum).astype(np.float32),
        sample_rate,
        ONSET_N_FFT
    )

    del onset_spectrum

    assessments = []
    notes_to_keep = {
        index: []
        for index in range(len(midi.instruments))
    }

    for event_index, item in enumerate(raw_notes):
        _, instrument_index, instrument, note = item

        note_data = make_note_data(
            event_index,
            instrument_index,
            instrument,
            note
        )

        assessment = assess_note(
            note_data,
            pitch_scores,
            onset_scores,
            sample_rate,
            HOP_LENGTH,
            audio_duration
        )

        assessments.append(assessment)

        if assessment["status"] != "removed":
            notes_to_keep[instrument_index].append(note)

    for instrument_index, instrument in enumerate(midi.instruments):
        instrument.notes = notes_to_keep[instrument_index]

    write_result(
        midi,
        CLEANED_MIDI_PATH,
        assessments
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Error en la limpieza MIDI: {error}",
            file=sys.stderr
        )
        raise