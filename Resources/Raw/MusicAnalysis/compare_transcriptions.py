"""Compara una transcripcion Magenta con Spotify Basic Pitch.

El script ejecuta Basic Pitch sobre el mismo audio, alinea las notas por
altura e inicio, y guarda MIDI alternativo, MIDI de consenso, JSON y CSV.
La comparacion es diagnostica: nunca borra notas de Magenta.

Variables de entorno requeridas:
  AUDIO_INPUT
  MAGENTA_MIDI_INPUT

Variables opcionales:
  BASIC_PITCH_MIDI_OUTPUT
  CONSENSUS_MIDI_OUTPUT
  COMPARISON_REPORT_OUTPUT
  COMPARISON_CSV_OUTPUT
  COMPARISON_MATCH_TOLERANCE_SECONDS (por defecto 0.12)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pretty_midi
import soundfile as sf


NOTE_NAMES = (
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
)


def midi_pitch_name(pitch: int) -> str:
    return f"{NOTE_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def finite(value: float | None, digits: int = 4):
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}.")
    return value


def _windowed_sinc_resample(samples, source_rate: int, target_rate: int):
    """Resamplea con sinc con ventana usando solo NumPy.

    Basic Pitch normalmente delega esta lectura a librosa. La lectura directa
    evita cargar backends de audio innecesarios en la app y mantiene la misma
    frecuencia de entrada que espera el modelo.
    """
    if source_rate == target_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)

    ratio = float(target_rate) / float(source_rate)
    output_length = max(1, int(round(samples.shape[0] * ratio)))
    radius = 24
    offsets = np.arange(-radius + 1, radius + 1, dtype=np.int64)
    cutoff = min(1.0, ratio)
    output = np.empty(output_length, dtype=np.float32)

    for block_start in range(0, output_length, 8192):
        block_end = min(output_length, block_start + 8192)
        output_indices = np.arange(block_start, block_end, dtype=np.float64)
        source_positions = output_indices / ratio
        centers = np.floor(source_positions).astype(np.int64)
        source_indices = centers[:, None] + offsets[None, :]
        distances = source_positions[:, None] - source_indices

        normalized_window_position = np.abs(distances) / float(radius)
        inside_window = normalized_window_position < 1.0
        window = np.zeros_like(distances, dtype=np.float64)
        x = normalized_window_position[inside_window]
        window[inside_window] = (
            0.42 + 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2.0 * np.pi * x)
        )
        weights = cutoff * np.sinc(cutoff * distances) * window

        valid = (source_indices >= 0) & (source_indices < samples.shape[0])
        weights *= valid
        weight_sums = np.sum(weights, axis=1)
        safe_indices = np.clip(source_indices, 0, samples.shape[0] - 1)
        selected = samples[safe_indices]
        weighted = np.sum(selected * weights, axis=1)
        nonzero = np.abs(weight_sums) > 1e-12
        block = np.zeros(block_end - block_start, dtype=np.float64)
        block[nonzero] = weighted[nonzero] / weight_sums[nonzero]
        output[block_start:block_end] = block.astype(np.float32)

    return output


def _load_audio_for_basic_pitch(audio_path, sr=22050, mono=True):
    print("BASIC_PITCH_STATUS: reading_wav", flush=True)
    waveform, source_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    if mono:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)
    elif waveform.shape[1] == 1:
        waveform = waveform[:, 0]

    if sr is not None and int(source_rate) != int(sr):
        print(
            f"BASIC_PITCH_STATUS: resampling_{source_rate}_to_{sr}",
            flush=True,
        )
        waveform = _windowed_sinc_resample(
            np.asarray(waveform, dtype=np.float32),
            int(source_rate),
            int(sr),
        )
        source_rate = int(sr)

    print(
        f"BASIC_PITCH_STATUS: audio_ready_samples_{len(waveform)}",
        flush=True,
    )
    return np.asarray(waveform, dtype=np.float32), int(source_rate)


def flatten_midi(midi: pretty_midi.PrettyMIDI):
    events = []
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note_index, note in enumerate(instrument.notes):
            start = float(note.start)
            end = float(note.end)
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            events.append({
                "eventIndex": len(events),
                "instrumentIndex": instrument_index,
                "instrumentName": instrument.name or f"Instrumento {instrument_index + 1}",
                "program": int(instrument.program),
                "sourceNoteIndex": note_index,
                "pitch": int(note.pitch),
                "pitchName": midi_pitch_name(note.pitch),
                "startSeconds": start,
                "endSeconds": end,
                "durationSeconds": end - start,
                "velocity": int(note.velocity),
            })
    events.sort(key=lambda event: (
        event["startSeconds"], event["pitch"], event["instrumentIndex"]
    ))
    for index, event in enumerate(events):
        event["eventIndex"] = index
    return events


def analyze_audio_start(waveform, sample_rate: int):
    """Mide el primer tramo activo sin tratarlo como nota musical."""
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or sample_rate <= 0:
        return {
            "activeStartSeconds": None,
            "leadingSilenceSeconds": None,
            "activityThresholdRms": None,
            "rmsAtStart": None,
        }

    frame_size = max(1, int(round(sample_rate * 0.01)))
    frame_count = int(math.ceil(len(audio) / frame_size))
    padded = np.pad(audio, (0, frame_count * frame_size - len(audio)))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    probe_count = max(1, min(frame_count, int(round(2.0 / 0.01))))
    noise_floor = float(np.percentile(rms[:probe_count], 20))
    threshold = max(0.006, noise_floor * 4.0)

    active_index = None
    for index in range(max(0, frame_count - 2)):
        if np.all(rms[index:index + 3] >= threshold):
            active_index = index
            break

    active_start = active_index * 0.01 if active_index is not None else None
    return {
        "activeStartSeconds": finite(active_start),
        "leadingSilenceSeconds": finite(active_start),
        "activityThresholdRms": finite(threshold, 6),
        "noiseFloorRms": finite(noise_floor, 6),
        "rmsAtStart": finite(float(rms[0]), 6),
        "audioDurationSeconds": finite(len(audio) / sample_rate),
    }


def _band_db(waveform, sample_rate: int, start_seconds: float, frequency: float):
    """Energía local de una banda estrecha; evidencia descriptiva solamente."""
    if frequency <= 0 or sample_rate <= 0:
        return None
    begin = max(0, int(round(start_seconds * sample_rate)))
    length = min(int(round(0.14 * sample_rate)), len(waveform) - begin)
    if length < max(32, int(sample_rate * 0.025)):
        return None
    segment = np.asarray(waveform[begin:begin + length], dtype=np.float64)
    segment = segment - float(np.mean(segment))
    segment *= np.hanning(length)
    fft_length = max(4096, 1 << int(math.ceil(math.log2(length))))
    spectrum = np.abs(np.fft.rfft(segment, n=fft_length))
    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate)
    half_width = max(4.0, frequency * 0.025)
    band = spectrum[np.abs(frequencies - frequency) <= half_width]
    if band.size == 0:
        return None
    magnitude = float(np.max(band)) / max(1.0, length / 2.0)
    return finite(20.0 * math.log10(max(magnitude, 1e-9)), 2)


def _quarter_seconds(bpm: float, tempo_unit: str) -> float:
    quarter_lengths = {
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eighth": 0.5,
        "sixteenth": 0.25,
        "32nd": 0.125,
    }
    return (60.0 / bpm) * quarter_lengths.get(tempo_unit.lower(), 1.0)


def score_position(start_seconds: float, audio_start_seconds: float):
    """Convierte el ataque a compás/pulso usando los ajustes aplicados en UI."""
    try:
        numerator = int(os.environ["SCORE_METER_NUMERATOR"])
        denominator = int(os.environ["SCORE_METER_DENOMINATOR"])
        bpm = float(os.environ["SCORE_TEMPO_BPM"])
        tempo_unit = os.environ.get("SCORE_TEMPO_UNIT", "quarter")
    except (KeyError, TypeError, ValueError):
        return None, None

    if numerator <= 0 or denominator <= 0 or bpm <= 0:
        return None, None
    seconds_per_quarter = _quarter_seconds(bpm, tempo_unit)
    beat_quarters = 4.0 / denominator
    bar_quarters = numerator * beat_quarters
    elapsed_quarters = max(
        0.0, (start_seconds - audio_start_seconds) / seconds_per_quarter
    )
    measure = int(math.floor(elapsed_quarters / bar_quarters)) + 1
    beat = (elapsed_quarters % bar_quarters) / beat_quarters + 1.0
    return measure, finite(beat, 3)


def octave_review_row(
    magenta_event,
    basic_event,
    offset,
    waveform,
    sample_rate,
    audio_start_seconds,
):
    aligned_basic_start = basic_event["startSeconds"] - offset
    aligned_start = (magenta_event["startSeconds"] + aligned_basic_start) / 2.0
    lower = min(magenta_event["pitch"], basic_event["pitch"])
    higher = max(magenta_event["pitch"], basic_event["pitch"])
    lower_frequency = 440.0 * (2.0 ** ((lower - 69.0) / 12.0))
    higher_frequency = 440.0 * (2.0 ** ((higher - 69.0) / 12.0))
    evidence_start = max(0.0, aligned_start + 0.025)
    lower_db = _band_db(waveform, sample_rate, evidence_start, lower_frequency)
    higher_db = _band_db(waveform, sample_rate, evidence_start, higher_frequency)
    measure, beat = score_position(aligned_start, audio_start_seconds)

    return {
        "startSeconds": finite(aligned_start),
        "measure": measure,
        "beat": beat,
        "magentaPitch": magenta_event["pitchName"],
        "magentaStartSeconds": finite(magenta_event["startSeconds"]),
        "basicPitchPitch": basic_event["pitchName"],
        "basicPitchStartSeconds": finite(basic_event["startSeconds"]),
        "startDeltaSeconds": finite(
            basic_event["startSeconds"] - magenta_event["startSeconds"] - offset
        ),
        "lowerPitch": midi_pitch_name(lower),
        "higherPitch": midi_pitch_name(higher),
        "lowerFundamentalDb": lower_db,
        "upperFrequencyDb": higher_db,
        "upperToLowerDb": (
            finite(higher_db - lower_db, 2)
            if lower_db is not None and higher_db is not None
            else None
        ),
        "review": (
            "Comparar ambas alturas con la partitura en el compás/pulso indicado. "
            "La energía de una banda aislada no distingue por sí sola un armónico "
            "de un doblaje real; no eliminar automáticamente."
        ),
    }


def write_octave_review_csv(path, rows):
    fields = [
        "startSeconds", "measure", "beat", "magentaPitch",
        "magentaStartSeconds", "basicPitchPitch", "basicPitchStartSeconds",
        "startDeltaSeconds", "lowerPitch", "higherPitch",
        "lowerFundamentalDb", "upperFrequencyDb", "upperToLowerDb", "review",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def estimate_global_start_offset(magenta_events, basic_events) -> float:
    """Estima una latencia global robusta entre los dos transcriptores.

    Solo usa la nota del mismo pitch mas cercana dentro de 450 ms. El limite
    evita que una coincidencia periodica distante desplace toda la alineacion.
    """
    basic_by_pitch = {}
    for event in basic_events:
        basic_by_pitch.setdefault(event["pitch"], []).append(event)

    offsets = []
    for magenta in magenta_events:
        candidates = basic_by_pitch.get(magenta["pitch"], ())
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda event: abs(event["startSeconds"] - magenta["startSeconds"]),
        )
        offset = nearest["startSeconds"] - magenta["startSeconds"]
        if abs(offset) <= 0.45:
            offsets.append(offset)

    if len(offsets) < 3:
        return 0.0

    # Recorta outliers antes de tomar la mediana para resistir notas repetidas.
    center = median(offsets)
    inliers = [value for value in offsets if abs(value - center) <= 0.12]
    if len(inliers) >= 3:
        center = median(inliers)
    return max(-0.25, min(0.25, float(center)))


def match_same_pitch(magenta_events, basic_events, offset, tolerance):
    """Empareja notas 1:1 con el mismo pitch y ataques cercanos."""
    possible_pairs = []
    for magenta_index, magenta in enumerate(magenta_events):
        expected_basic_start = magenta["startSeconds"] + offset
        for basic_index, basic in enumerate(basic_events):
            if magenta["pitch"] != basic["pitch"]:
                continue
            delta = basic["startSeconds"] - expected_basic_start
            if abs(delta) > tolerance:
                continue
            duration_delta = abs(
                magenta["durationSeconds"] - basic["durationSeconds"]
            )
            cost = abs(delta) + 0.15 * min(duration_delta, 0.5)
            possible_pairs.append((cost, abs(delta), magenta_index, basic_index, delta))

    possible_pairs.sort()
    used_magenta = set()
    used_basic = set()
    matches = []
    for _, _, magenta_index, basic_index, delta in possible_pairs:
        if magenta_index in used_magenta or basic_index in used_basic:
            continue
        used_magenta.add(magenta_index)
        used_basic.add(basic_index)
        matches.append({
            "magentaIndex": magenta_index,
            "basicPitchIndex": basic_index,
            "startDeltaSeconds": delta,
        })
    matches.sort(key=lambda item: item["magentaIndex"])
    return matches, used_magenta, used_basic


def find_octave_counterpart(
    event,
    other_events,
    other_used,
    offset,
    tolerance,
    event_is_magenta,
):
    """Busca discrepancia de octava sin confundirla con acuerdo exacto."""
    expected_other_start = event["startSeconds"] + (offset if event_is_magenta else -offset)
    candidates = []
    for index, other in enumerate(other_events):
        if index in other_used:
            continue
        if abs(int(other["pitch"]) - int(event["pitch"])) != 12:
            continue
        delta = other["startSeconds"] - expected_other_start
        if abs(delta) <= tolerance * 1.35:
            candidates.append((abs(delta), index, delta))
    if not candidates:
        return None
    _, index, delta = min(candidates)
    return index, delta


def match_octave_events(
    magenta_events,
    basic_events,
    used_magenta,
    used_basic,
    offset,
    tolerance,
):
    """Empareja discrepancias de octava 1:1 después de los acuerdos exactos."""
    possible = []
    for magenta_index, magenta in enumerate(magenta_events):
        if magenta_index in used_magenta:
            continue
        for basic_index, basic in enumerate(basic_events):
            if basic_index in used_basic:
                continue
            if abs(int(magenta["pitch"]) - int(basic["pitch"])) != 12:
                continue
            delta = basic["startSeconds"] - (
                magenta["startSeconds"] + offset
            )
            if abs(delta) <= tolerance * 1.35:
                possible.append((abs(delta), magenta_index, basic_index, delta))

    possible.sort()
    octave_by_magenta = {}
    octave_by_basic = {}
    for _, magenta_index, basic_index, delta in possible:
        if (magenta_index in octave_by_magenta or
                basic_index in octave_by_basic):
            continue
        octave_by_magenta[magenta_index] = (basic_index, delta)
        octave_by_basic[basic_index] = (magenta_index, delta)
    return octave_by_magenta, octave_by_basic


def build_consensus_midi(magenta_midi, magenta_events, basic_events, matches, path):
    """Guarda solo los ataques del mismo pitch reconocidos por ambos modelos."""
    consensus = pretty_midi.PrettyMIDI()
    instruments = {}
    for instrument_index, instrument in enumerate(magenta_midi.instruments):
        if instrument.is_drum:
            continue
        instruments[instrument_index] = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=False,
            name=instrument.name or f"Instrumento {instrument_index + 1}",
        )

    for match in matches:
        magenta = magenta_events[match["magentaIndex"]]
        basic = basic_events[match["basicPitchIndex"]]
        instrument = instruments.get(magenta["instrumentIndex"])
        if instrument is None:
            continue
        # Se conserva el tiempo de Magenta para que el consenso siga alineado
        # con la transcripcion principal que consume el resto de la aplicacion.
        instrument.notes.append(pretty_midi.Note(
            velocity=magenta["velocity"],
            pitch=magenta["pitch"],
            start=magenta["startSeconds"],
            end=max(magenta["startSeconds"] + 0.025, magenta["endSeconds"]),
        ))

    for instrument in instruments.values():
        instrument.notes.sort(key=lambda note: (note.start, note.pitch))
        if instrument.notes:
            consensus.instruments.append(instrument)
    consensus.write(path)


def write_csv(path, magenta_rows, basic_rows):
    fields = [
        "source", "eventIndex", "pitchName", "startSeconds", "durationSeconds",
        "velocity", "status", "counterpartPitchName", "counterpartStartSeconds",
        "startDeltaSeconds", "explanation",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in magenta_rows + basic_rows:
            writer.writerow({key: row.get(key) for key in fields})


def run():
    audio_path = Path(require_env("AUDIO_INPUT"))
    magenta_path = Path(require_env("MAGENTA_MIDI_INPUT"))
    if not audio_path.is_file():
        raise FileNotFoundError(f"No se encontro el audio: {audio_path}")
    if not magenta_path.is_file():
        raise FileNotFoundError(f"No se encontro el MIDI de Magenta: {magenta_path}")

    output_dir = Path(os.environ.get("COMPARISON_OUTPUT_DIRECTORY", str(magenta_path.parent)))
    output_dir.mkdir(parents=True, exist_ok=True)
    basic_midi_path = Path(os.environ.get(
        "BASIC_PITCH_MIDI_OUTPUT",
        str(output_dir / "grabacion_basic_pitch.mid"),
    ))
    consensus_midi_path = Path(os.environ.get(
        "CONSENSUS_MIDI_OUTPUT",
        str(output_dir / "grabacion_consenso.mid"),
    ))
    report_path = Path(os.environ.get(
        "COMPARISON_REPORT_OUTPUT",
        str(output_dir / "transcription_comparison.json"),
    ))
    csv_path = Path(os.environ.get(
        "COMPARISON_CSV_OUTPUT",
        str(output_dir / "transcription_comparison.csv"),
    ))
    octave_review_path = Path(os.environ.get(
        "OCTAVE_REVIEW_CSV_OUTPUT",
        str(output_dir / "octave_discrepancy_review.csv"),
    ))
    for path in (basic_midi_path, consensus_midi_path, report_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    octave_review_path.parent.mkdir(parents=True, exist_ok=True)

    # Spotify Basic Pitch es un modelo independiente de Magenta. Se usa su API
    # oficial y se conserva su transcripcion completa como un archivo separado.
    print("BASIC_PITCH_STATUS: loading_model", flush=True)
    import basic_pitch.inference as basic_pitch_inference

    # No se cambia el modelo ni el decoder de Basic Pitch; solo la lectura de
    # WAV para hacer mono y remuestrear a 22.05 kHz antes de su inferencia.
    basic_pitch_inference.librosa.load = _load_audio_for_basic_pitch

    print("BASIC_PITCH_STATUS: transcribing_audio", flush=True)
    minimum_note_length_ms = float(os.environ.get(
        "BASIC_PITCH_MINIMUM_NOTE_LENGTH_MS", "45.0"
    ))
    onnx_model_path = (
        Path(basic_pitch_inference.__file__).resolve().parent
        / "saved_models"
        / "icassp_2022"
        / "nmp.onnx"
    )
    if onnx_model_path.is_file() and basic_pitch_inference.ONNX_PRESENT:
        # En Windows, Basic Pitch incluye ONNX. Si TensorFlow tambien esta
        # instalado, su selector prefiere TensorFlow; elegimos ONNX para evitar
        # cargar el SavedModel mas pesado en cada grabacion.
        basic_pitch_inference.TF_PRESENT = False
        basic_pitch_inference.TFLITE_PRESENT = False
        ort = basic_pitch_inference.ort
        original_inference_session = ort.InferenceSession

        def bounded_inference_session(path, *args, **kwargs):
            session_options = kwargs.pop("sess_options", None)
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = min(
                    4, max(1, os.cpu_count() or 1)
                )
                session_options.inter_op_num_threads = 1
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return original_inference_session(
                path,
                *args,
                sess_options=session_options,
                **kwargs,
            )

        ort.InferenceSession = bounded_inference_session
        model = basic_pitch_inference.Model(str(onnx_model_path))
        ort.InferenceSession = original_inference_session
        _, basic_midi, _ = basic_pitch_inference.predict(
            audio_path=str(audio_path),
            model_or_model_path=model,
            minimum_note_length=minimum_note_length_ms,
        )
    else:
        _, basic_midi, _ = basic_pitch_inference.predict(
            audio_path=str(audio_path),
            minimum_note_length=minimum_note_length_ms,
        )
    if basic_midi is None:
        raise RuntimeError("Basic Pitch no devolvio una transcripcion MIDI.")
    print("BASIC_PITCH_STATUS: saving_midi", flush=True)
    basic_midi.write(str(basic_midi_path))

    analysis_waveform, analysis_sample_rate = _load_audio_for_basic_pitch(
        audio_path
    )
    audio_start_metrics = analyze_audio_start(
        analysis_waveform, analysis_sample_rate
    )
    audio_active_start = audio_start_metrics["activeStartSeconds"]
    if audio_active_start is None:
        audio_active_start = 0.0

    magenta_midi = pretty_midi.PrettyMIDI(str(magenta_path))
    magenta_events = flatten_midi(magenta_midi)
    basic_events = flatten_midi(basic_midi)
    offset = estimate_global_start_offset(magenta_events, basic_events)
    tolerance = float(os.environ.get(
        "COMPARISON_MATCH_TOLERANCE_SECONDS", "0.12"
    ))
    tolerance = max(0.04, min(0.25, tolerance))

    matches, used_magenta, used_basic = match_same_pitch(
        magenta_events, basic_events, offset, tolerance
    )
    magenta_rows = []
    basic_rows = []
    octave_conflicts_magenta = 0
    octave_conflicts_basic = 0
    octave_pairs = []
    magenta_match_by_index = {
        item["magentaIndex"]: item["basicPitchIndex"] for item in matches
    }
    basic_match_by_index = {
        item["basicPitchIndex"]: item["magentaIndex"] for item in matches
    }
    octave_by_magenta, octave_by_basic = match_octave_events(
        magenta_events,
        basic_events,
        used_magenta,
        used_basic,
        offset,
        tolerance,
    )

    for index, event in enumerate(magenta_events):
        common = {
            "source": "Magenta",
            "eventIndex": index,
            "pitchName": event["pitchName"],
            "startSeconds": finite(event["startSeconds"]),
            "durationSeconds": finite(event["durationSeconds"]),
            "velocity": event["velocity"],
        }
        if index in magenta_match_by_index:
            basic_index = magenta_match_by_index[index]
            counterpart = basic_events[basic_index]
            delta = counterpart["startSeconds"] - event["startSeconds"] - offset
            magenta_rows.append({
                **common,
                "status": "agreement",
                "counterpartPitchName": counterpart["pitchName"],
                "counterpartStartSeconds": finite(counterpart["startSeconds"]),
                "startDeltaSeconds": finite(delta),
                "explanation": "Ambos modelos detectaron la misma altura con ataque cercano.",
            })
            continue

        octave_match = octave_by_magenta.get(index)
        if octave_match is not None:
            basic_index, delta = octave_match
            counterpart = basic_events[basic_index]
            octave_conflicts_magenta += 1
            octave_pairs.append((index, basic_index))
            status = "octave_disagreement"
            explanation = (
                "Basic Pitch detecto una nota a una octava de distancia. "
                "Puede ser un armonico o una diferencia real entre modelos; revisar."
            )
        else:
            counterpart = None
            delta = None
            status = "magenta_only"
            explanation = (
                "Solo Magenta detecto esta nota. Puede ser una nota omitida por "
                "Basic Pitch; no se elimina automaticamente."
            )
        magenta_rows.append({
            **common,
            "status": status,
            "counterpartPitchName": counterpart["pitchName"] if counterpart else None,
            "counterpartStartSeconds": finite(counterpart["startSeconds"]) if counterpart else None,
            "startDeltaSeconds": finite(delta),
            "explanation": explanation,
        })

    for index, event in enumerate(basic_events):
        common = {
            "source": "BasicPitch",
            "eventIndex": index,
            "pitchName": event["pitchName"],
            "startSeconds": finite(event["startSeconds"]),
            "durationSeconds": finite(event["durationSeconds"]),
            "velocity": event["velocity"],
        }
        if index in basic_match_by_index:
            magenta_index = basic_match_by_index[index]
            counterpart = magenta_events[magenta_index]
            delta = event["startSeconds"] - counterpart["startSeconds"] - offset
            basic_rows.append({
                **common,
                "status": "agreement",
                "counterpartPitchName": counterpart["pitchName"],
                "counterpartStartSeconds": finite(counterpart["startSeconds"]),
                "startDeltaSeconds": finite(delta),
                "explanation": "Ambos modelos detectaron la misma altura con ataque cercano.",
            })
            continue

        octave_match = octave_by_basic.get(index)
        if octave_match is not None:
            magenta_index, delta = octave_match
            counterpart = magenta_events[magenta_index]
            octave_conflicts_basic += 1
            octave_pairs.append((magenta_index, index))
            status = "octave_disagreement"
            explanation = (
                "Magenta detecto una nota a una octava de distancia. "
                "Puede ser un armonico o una diferencia real entre modelos; revisar."
            )
        else:
            counterpart = None
            delta = None
            status = "basic_pitch_only"
            explanation = (
                "Solo Basic Pitch detecto esta nota. Es una candidata para revisar "
                "y no se agrega automaticamente a la transcripcion principal."
            )
        basic_rows.append({
            **common,
            "status": status,
            "counterpartPitchName": counterpart["pitchName"] if counterpart else None,
            "counterpartStartSeconds": finite(counterpart["startSeconds"]) if counterpart else None,
            "startDeltaSeconds": finite(delta),
            "explanation": explanation,
        })

    build_consensus_midi(
        magenta_midi, magenta_events, basic_events, matches,
        str(consensus_midi_path),
    )

    unique_octave_pairs = []
    seen_octave_pairs = set()
    for magenta_index, basic_index in octave_pairs:
        pair = (magenta_index, basic_index)
        if pair in seen_octave_pairs:
            continue
        seen_octave_pairs.add(pair)
        unique_octave_pairs.append(octave_review_row(
            magenta_events[magenta_index],
            basic_events[basic_index],
            offset,
            analysis_waveform,
            analysis_sample_rate,
            audio_active_start,
        ))
    unique_octave_pairs.sort(key=lambda row: (
        row["startSeconds"] if row["startSeconds"] is not None else -1.0
    ))

    magenta_count = len(magenta_events)
    basic_count = len(basic_events)
    agreement_count = len(matches)
    report = {
        "version": "1.1",
        "status": "completed",
        "audioPath": str(audio_path),
        "magentaMidiPath": str(magenta_path),
        "basicPitchMidiPath": str(basic_midi_path),
        "consensusMidiPath": str(consensus_midi_path),
        "csvPath": str(csv_path),
        "method": {
            "magenta": "Onsets and Frames",
            "independentTranscriber": "Spotify Basic Pitch",
            "matching": "same MIDI pitch and onset within tolerance after robust global offset correction",
            "basicPitchMinimumNoteLengthMs": finite(minimum_note_length_ms),
            "octaveReview": "audio frequency evidence plus score position; octave conflicts are never deleted automatically",
            "automaticDeletion": False,
        },
        "alignment": {
            "estimatedBasicPitchMinusMagentaSeconds": finite(offset),
            "matchToleranceSeconds": finite(tolerance),
        },
        "summary": {
            "magentaNotes": magenta_count,
            "basicPitchNotes": basic_count,
            "samePitchAgreement": agreement_count,
            "magentaOnly": sum(1 for row in magenta_rows if row["status"] == "magenta_only"),
            "magentaOctaveDisagreement": octave_conflicts_magenta,
            "basicPitchOnly": sum(1 for row in basic_rows if row["status"] == "basic_pitch_only"),
            "basicPitchOctaveDisagreement": octave_conflicts_basic,
            "magentaAgreementPercent": finite(100.0 * agreement_count / magenta_count) if magenta_count else 0.0,
            "basicPitchAgreementPercent": finite(100.0 * agreement_count / basic_count) if basic_count else 0.0,
        },
        "audioStart": {
            **audio_start_metrics,
            "firstMagentaNoteSeconds": finite(
                magenta_events[0]["startSeconds"] if magenta_events else None
            ),
            "firstBasicPitchNoteSeconds": finite(
                basic_events[0]["startSeconds"] if basic_events else None
            ),
            "magentaGapAfterAudioActivitySeconds": finite(
                max(
                    0.0,
                    magenta_events[0]["startSeconds"] - audio_active_start,
                ) if magenta_events else None
            ),
            "note": (
                "Si el audio tiene actividad antes de la primera nota MIDI, "
                "revisar start_boundary_diagnostic.json y el MIDI candidato "
                "grabacion_inicio_revisado.mid."
            ),
        },
        "octaveReview": {
            "csvPath": str(octave_review_path),
            "uniquePairs": len(unique_octave_pairs),
            "scoreSettings": {
                "numerator": os.environ.get("SCORE_METER_NUMERATOR"),
                "denominator": os.environ.get("SCORE_METER_DENOMINATOR"),
                "tempoBpm": os.environ.get("SCORE_TEMPO_BPM"),
                "tempoUnit": os.environ.get("SCORE_TEMPO_UNIT", "quarter"),
            },
            "interpretation": (
                "La tabla ubica cada discrepancia en el compás/pulso estimado y "
                "muestra energía local en ambas alturas. La partitura de imagen "
                "no se interpreta automáticamente; confirme cada caso en la "
                "partitura antes de cambiar notas."
            ),
        },
        "interpretation": (
            "El acuerdo entre modelos aumenta el respaldo de una nota. "
            "Una nota detectada por un solo modelo queda como candidata de revision: "
            "la ausencia en el otro modelo no demuestra que sea falsa. "
            "Las discrepancias de octava son ambiguas y no se borran automaticamente."
        ),
        "magentaEvents": magenta_rows,
        "basicPitchEvents": basic_rows,
    }

    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    write_csv(csv_path, magenta_rows, basic_rows)
    write_octave_review_csv(octave_review_path, unique_octave_pairs)

    print("COMPARISON_SUMMARY:" + json.dumps(report["summary"], ensure_ascii=False))
    print("COMPARISON_REPORT:" + str(report_path))
    print("OCTAVE_REVIEW_CSV:" + str(octave_review_path))
    print("BASIC_PITCH_MIDI:" + str(basic_midi_path))
    print("CONSENSUS_MIDI:" + str(consensus_midi_path))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
