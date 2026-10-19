import json
import math
import os
import sys
from collections import defaultdict

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

# Eliminación automática conservadora. Además de notas con evidencia tonal
# extremadamente baja, se permite retirar un artefacto breve y débil cuando
# la misma energía tonal ya estaba presente antes del inicio MIDI.
AUTO_REMOVE_MAX_PITCH_DB = -15.0
AUTO_REMOVE_MAX_ONSET_DB = -12.0
AUTO_REMOVE_MIN_DURATION_SECONDS = 0.10
AUTO_REMOVE_MAX_EVIDENCE_SCORE = 8.0
AUTO_REMOVE_MAX_SUSTAIN_COVERAGE = 0.12

NOISE_MAX_DURATION_SECONDS = 0.12
NOISE_MAX_VELOCITY = 48
NOISE_MAX_EVIDENCE_SCORE = 70.0
NOISE_MAX_PITCH_CONTRAST_DB = 3.0
NOISE_MAX_ONSET_RISE_DB = 6.0
NOISE_MAX_PRE_ONSET_RISE_DB = 6.0

SUPPORTED_MIN_EVIDENCE_SCORE = 72.0
SUPPORTED_MIN_CONTRAST_DB = 0.0
SUPPORTED_MIN_ONSET_DB = 0.0
SUPPORTED_MIN_SUSTAIN_COVERAGE = 0.35
OCTAVE_AMBIGUITY_PENALTY = 8.0

EVIDENCE_WEIGHTS = {
    "pitchContrast": 0.38,
    "onsetRise": 0.32,
    "sustainCoverage": 0.15,
    "preOnsetRise": 0.15,
}

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


def build_pitch_onset_salience(magnitude, sample_rate, n_fft):
    """Separa ataques nuevos del nivel sostenido de cada armónico."""
    positive_flux = np.maximum(
        np.diff(
            magnitude,
            axis=1,
            prepend=magnitude[:, :1],
        ),
        0.0,
    )

    return build_pitch_salience(
        positive_flux,
        sample_rate,
        n_fft,
    )


def median_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(np.median(values))


def _map_db_to_score(value, low_db=-12.0, high_db=18.0):
    """Convierte una señal en dB a una escala lineal de evidencia 0..100."""
    if value is None or not math.isfinite(float(value)):
        return None

    normalized = (float(value) - low_db) / (high_db - low_db)
    return max(0.0, min(100.0, normalized * 100.0))


def calculate_evidence_score(
    pitch_contrast_db,
    onset_rise_db,
    sustain_coverage,
    pre_onset_pitch_rise_db,
    octave_overlap_risk=False,
):
    """Puntúa evidencia relativa; no es una probabilidad de que la nota sea real."""
    components = {}
    if pitch_contrast_db is not None:
        components["pitchContrast"] = _map_db_to_score(
            pitch_contrast_db,
            low_db=-12.0,
            high_db=12.0,
        )
    if onset_rise_db is not None:
        components["onsetRise"] = _map_db_to_score(
            onset_rise_db,
            low_db=-12.0,
            high_db=18.0,
        )
    if sustain_coverage is not None:
        components["sustainCoverage"] = max(
            0.0,
            min(100.0, float(sustain_coverage) * 100.0),
        )
    if pre_onset_pitch_rise_db is not None:
        components["preOnsetRise"] = _map_db_to_score(
            pre_onset_pitch_rise_db,
            low_db=-9.0,
            high_db=12.0,
        )

    available = [
        (name, value, EVIDENCE_WEIGHTS[name])
        for name, value in components.items()
        if value is not None
    ]
    total_weight = sum(weight for _, _, weight in available)
    if total_weight <= 0:
        return None, components

    score = sum(
        value * weight
        for _, value, weight in available
    ) / total_weight

    if octave_overlap_risk:
        score = max(0.0, score - OCTAVE_AMBIGUITY_PENALTY)

    return max(0.0, min(100.0, score)), components


def classify_note_evidence(
    evidence_score,
    pitch_contrast_db,
    onset_rise_db,
    sustain_coverage,
    pre_onset_pitch_rise_db,
    duration_seconds,
    velocity,
    low_velocity_cutoff,
    octave_overlap_risk=False,
):
    """Aplica cortes conservadores y devuelve la etiqueta y explicación."""
    available_count = sum(
        value is not None
        for value in (
            pitch_contrast_db,
            onset_rise_db,
            sustain_coverage,
            pre_onset_pitch_rise_db,
        )
    )

    if available_count < 2 or evidence_score is None:
        return "inconclusive", (
            "No hay dos señales medibles suficientes para puntuar; "
            "se conserva."
        )

    # Un tono estacionario (por ejemplo, zumbido de 50/60 Hz) puede parecer
    # una nota sostenida. Solo se borra si además es muy corta, débil, poco
    # contrastante y de velocidad atípicamente baja para esta interpretación.
    stationary_low_note = (
        duration_seconds <= NOISE_MAX_DURATION_SECONDS
        and velocity <= low_velocity_cutoff
        and evidence_score <= NOISE_MAX_EVIDENCE_SCORE
        and pitch_contrast_db is not None
        and pitch_contrast_db <= NOISE_MAX_PITCH_CONTRAST_DB
        and onset_rise_db is not None
        and onset_rise_db <= NOISE_MAX_ONSET_RISE_DB
        and pre_onset_pitch_rise_db is not None
        and pre_onset_pitch_rise_db <= NOISE_MAX_PRE_ONSET_RISE_DB
        and not octave_overlap_risk
    )
    if stationary_low_note:
        return "removed", (
            "Evento breve y de velocidad atípicamente baja; la energía de "
            "esa altura ya estaba presente antes del inicio y no muestra "
            "un ataque claro. Se elimina solo de la copia MIDI."
        )

    strict_low_evidence = (
        duration_seconds >= AUTO_REMOVE_MIN_DURATION_SECONDS
        and evidence_score <= AUTO_REMOVE_MAX_EVIDENCE_SCORE
        and pitch_contrast_db is not None
        and pitch_contrast_db <= AUTO_REMOVE_MAX_PITCH_DB
        and onset_rise_db is not None
        and onset_rise_db <= AUTO_REMOVE_MAX_ONSET_DB
        and sustain_coverage is not None
        and sustain_coverage <= AUTO_REMOVE_MAX_SUSTAIN_COVERAGE
        and pre_onset_pitch_rise_db is not None
        and pre_onset_pitch_rise_db <= 0.0
        and not octave_overlap_risk
    )
    if strict_low_evidence:
        return "removed", (
            "Evidencia extremadamente baja y consistente en altura, "
            "ataque y duración; se elimina solo de la copia MIDI."
        )

    strong_evidence = (
        evidence_score >= SUPPORTED_MIN_EVIDENCE_SCORE
        and pitch_contrast_db is not None
        and pitch_contrast_db >= SUPPORTED_MIN_CONTRAST_DB
        and onset_rise_db is not None
        and onset_rise_db >= SUPPORTED_MIN_ONSET_DB
        and (
            sustain_coverage is None
            or sustain_coverage >= SUPPORTED_MIN_SUSTAIN_COVERAGE
        )
        and not octave_overlap_risk
    )
    if strong_evidence:
        return "supported", (
            "Contraste de altura, ataque y duración aportan evidencia "
            "coherente; se conserva."
        )

    if octave_overlap_risk:
        return "review", (
            "La nota se solapa con una nota 12 semitonos más grave; sus "
            "armónicos podrían explicar parte de la energía. También puede "
            "ser un doblaje real, por lo que no se elimina."
        )

    return "review", (
        "La evidencia es débil o contradictoria y no cumple el umbral "
        "estricto de eliminación; se conserva para revisión."
    )


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


def calculate_sustain_coverage(
    pitch_scores,
    pitch,
    event_frames,
):
    """Mide qué parte de la nota mantiene energía tonal durante el evento."""
    if event_frames.size < 3:
        return None

    target = np.asarray(
        pitch_scores[pitch, event_frames],
        dtype=float,
    )
    if target.size == 0:
        return None

    neighbor_rows = []
    for neighbor_pitch in (pitch - 2, pitch - 1, pitch + 1, pitch + 2):
        if 0 <= neighbor_pitch <= 127:
            neighbor_rows.append(
                pitch_scores[neighbor_pitch, event_frames]
            )

    if not neighbor_rows:
        return None

    neighbor_floor = np.median(
        np.asarray(neighbor_rows, dtype=float),
        axis=0,
    )
    peak = float(np.max(target))
    if not math.isfinite(peak) or peak <= EPSILON:
        return 0.0

    # Exige que la altura supere tanto el piso de sus vecinas como una parte
    # significativa de su propio pico. El umbral anterior (8% del pico y
    # 0.70x las vecinas) hacía que casi cualquier energía residual contara
    # como cobertura completa.
    threshold = np.maximum(neighbor_floor * 1.40, peak * 0.20)
    return float(np.mean(target >= threshold))


def calculate_pre_onset_pitch_rise(
    pitch_scores,
    pitch,
    start_seconds,
    end_seconds,
    sample_rate,
    hop_length,
):
    """Compara la energía tonal del evento con la presente antes del ataque.

    Un zumbido estacionario puede parecer una nota porque ocupa una banda
    tonal. Si esa energía ya estaba antes del MIDI y no crece con el evento,
    esta medida lo señala como evidencia débil.
    """
    frame_count = pitch_scores.shape[1]
    before_frames = frames_between(
        start_seconds - 0.16,
        start_seconds - 0.04,
        sample_rate,
        hop_length,
        frame_count,
    )

    after_end = min(start_seconds + 0.16, end_seconds - 0.005)
    after_frames = frames_between(
        start_seconds + 0.02,
        after_end,
        sample_rate,
        hop_length,
        frame_count,
    )

    if before_frames.size < 2 or after_frames.size < 2:
        return None

    before_level = median_or_none(pitch_scores[pitch, before_frames])
    after_level = median_or_none(pitch_scores[pitch, after_frames])
    return ratio_to_db(after_level, before_level)


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


def find_octave_overlap_risks(raw_notes):
    """Marca notas altas solapadas con la misma nota una octava abajo.

    Es una advertencia de ambigüedad, no prueba de error: los doblajes de
    octava son comunes en piano y nunca se eliminan solo por esta regla.
    """
    by_pitch = defaultdict(list)
    for event_index, item in enumerate(raw_notes):
        _, instrument_index, instrument, midi_note = item
        if instrument.is_drum:
            continue
        by_pitch[int(midi_note.pitch)].append((event_index, item))

    risks = set()
    for event_index, item in enumerate(raw_notes):
        _, _, instrument, midi_note = item
        if instrument.is_drum or int(midi_note.pitch) < 12:
            continue

        lower_pitch = int(midi_note.pitch) - 12
        upper_start = float(midi_note.start)
        upper_end = float(midi_note.end)
        upper_duration = upper_end - upper_start
        if upper_duration <= 0:
            continue

        for lower_index, lower_item in by_pitch.get(lower_pitch, []):
            if lower_index == event_index:
                continue

            _, _, lower_instrument, lower_note = lower_item
            if lower_instrument.is_drum:
                continue

            lower_start = float(lower_note.start)
            lower_end = float(lower_note.end)
            lower_duration = lower_end - lower_start
            if lower_duration <= 0:
                continue

            overlap = min(upper_end, lower_end) - max(upper_start, lower_start)
            shorter_duration = min(upper_duration, lower_duration)
            overlap_ratio = overlap / shorter_duration

            # El armónico puede aparecer después del ataque grave: el MIDI
            # a veces separa su inicio unos milisegundos o incluso una
            # subdivisión. Marcamos como ambiguo si la nota inferior ya
            # estaba activa al entrar la superior y cubre la mayor parte de
            # esta. Nunca se elimina solo por esta coincidencia.
            if (
                lower_start <= upper_start + 0.07
                and overlap_ratio >= 0.65
            ):
                risks.add(event_index)
                break

    return risks


def estimate_low_velocity_cutoff(raw_notes):
    """Estima un límite robusto para detectar eventos muy débiles.

    El límite se acota para que una interpretación con dinámica amplia no
    convierta notas pianissimo en ruido automáticamente.
    """
    velocities = [
        int(item[3].velocity)
        for item in raw_notes
        if not item[2].is_drum
    ]
    if not velocities:
        return NOISE_MAX_VELOCITY

    median_velocity = float(np.median(velocities))
    mad = float(np.median(
        np.abs(np.asarray(velocities, dtype=float) - median_velocity)
    ))
    adaptive_cutoff = median_velocity - max(18.0, 3.0 * 1.4826 * mad)
    return int(max(28.0, min(NOISE_MAX_VELOCITY, adaptive_cutoff)))


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
    audio_duration,
    low_velocity_cutoff,
    octave_overlap_risk=False,
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
            "preOnsetPitchRiseDb": None,
            "sustainCoverage": None,
            "evidenceScore": None,
            "octaveOverlapRisk": False,
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
            "preOnsetPitchRiseDb": None,
            "sustainCoverage": None,
            "evidenceScore": None,
            "octaveOverlapRisk": bool(octave_overlap_risk),
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

    pre_onset_pitch_rise_db = calculate_pre_onset_pitch_rise(
        pitch_scores,
        pitch,
        start,
        end,
        sample_rate,
        hop_length,
    )

    sustain_coverage = calculate_sustain_coverage(
        pitch_scores,
        pitch,
        event_frames,
    )

    evidence_score, _ = calculate_evidence_score(
        pitch_contrast_db,
        onset_rise_db,
        sustain_coverage,
        pre_onset_pitch_rise_db,
        octave_overlap_risk,
    )

    status, reason = classify_note_evidence(
        evidence_score,
        pitch_contrast_db,
        onset_rise_db,
        sustain_coverage,
        pre_onset_pitch_rise_db,
        duration,
        int(note_data["velocity"]),
        low_velocity_cutoff,
        octave_overlap_risk,
    )

    details = []
    if pitch_contrast_db is not None:
        details.append(f"contraste {pitch_contrast_db:.1f} dB")
    if onset_rise_db is not None:
        details.append(f"ataque {onset_rise_db:.1f} dB")
    if pre_onset_pitch_rise_db is not None:
        details.append(f"crecimiento tonal local {pre_onset_pitch_rise_db:.1f} dB")
    if sustain_coverage is not None:
        details.append(f"cobertura {sustain_coverage:.0%}")

    if evidence_score is not None:
        reason += (
            f" Puntaje de evidencia: {evidence_score:.1f}/100 "
            "(no es probabilidad)"
        )
    if details:
        reason += "; " + ", ".join(details) + "."

    return {
        "note": note_data,
        "pitchContrastDb": finite_or_none(pitch_contrast_db),
        "onsetRiseDb": finite_or_none(onset_rise_db),
        "preOnsetPitchRiseDb": finite_or_none(pre_onset_pitch_rise_db),
        "sustainCoverage": finite_or_none(sustain_coverage),
        "evidenceScore": finite_or_none(evidence_score, 1),
        "octaveOverlapRisk": bool(octave_overlap_risk),
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

    octave_overlap_risks = find_octave_overlap_risks(raw_notes)
    low_velocity_cutoff = estimate_low_velocity_cutoff(raw_notes)

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

    onset_magnitude = np.abs(onset_spectrum).astype(np.float32)
    onset_scores = build_pitch_onset_salience(
        onset_magnitude,
        sample_rate,
        ONSET_N_FFT
    )

    del onset_magnitude
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
            audio_duration,
            low_velocity_cutoff,
            event_index in octave_overlap_risks,
        )

        assessments.append(assessment)

        if assessment["status"] != "removed":
            notes_to_keep[instrument_index].append(note)

    for instrument_index, instrument in enumerate(midi.instruments):
        instrument.notes = notes_to_keep[instrument_index]

    scored_values = [
        item["evidenceScore"]
        for item in assessments
        if item["evidenceScore"] is not None
    ]
    score_median = median_or_none(scored_values)
    octave_risk_count = sum(
        1 for item in assessments
        if item["octaveOverlapRisk"]
    )
    print(
        "RESUMEN PUNTUACIÓN:",
        f"mediana={finite_or_none(score_median, 1)} / 100;",
        f"riesgo de octava={octave_risk_count};",
        f"límite adaptativo de velocidad baja={low_velocity_cutoff};",
        "los riesgos de octava se conservan para revisión.",
        flush=True,
    )

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
