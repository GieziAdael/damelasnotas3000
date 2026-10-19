"""Detecta omisiones al comienzo y recupera candidatos con contexto previo.

Se invoca desde MagentaService después de la transcripción inicial. Solo pide
una segunda pasada cuando el WAV contiene audio desde el inicio y Magenta no
produce notas hasta mucho después. La salida corregida conserva el MIDI base y
añade exclusivamente candidatos anteriores al primer ataque original.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf


def _audio_activity(audio_path: Path, frame_seconds: float = 0.01) -> dict:
    audio, sample_rate = sf.read(
        str(audio_path), dtype="float32", always_2d=True
    )
    if audio.size == 0 or sample_rate <= 0:
        raise ValueError("El WAV no contiene audio legible.")

    mono = np.mean(audio, axis=1, dtype=np.float32)
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(math.ceil(len(mono) / frame_size))
    padded = np.pad(mono, (0, frame_count * frame_size - len(mono)))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    probe_frames = max(1, min(frame_count, int(round(2.0 / frame_seconds))))
    noise_floor = float(np.percentile(rms[:probe_frames], 20))
    threshold = max(0.006, noise_floor * 4.0)

    # Exige 30 ms consecutivos por encima del umbral para ignorar clics aislados.
    active_index = None
    required_frames = 3
    for index in range(max(0, frame_count - required_frames + 1)):
        if np.all(rms[index:index + required_frames] >= threshold):
            active_index = index
            break

    active_start = (
        active_index * frame_seconds if active_index is not None else None
    )
    return {
        "sampleRate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "durationSeconds": float(len(mono) / sample_rate),
        "activeStartSeconds": active_start,
        "leadingSilenceSeconds": active_start,
        "noiseFloorRms": noise_floor,
        "activityThresholdRms": threshold,
        "rmsAtStart": float(rms[0]) if len(rms) else 0.0,
        "audio": audio,
    }


def _first_note_seconds(midi: pretty_midi.PrettyMIDI) -> float | None:
    starts = [
        float(note.start)
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
        if math.isfinite(float(note.start))
    ]
    return min(starts) if starts else None


def analyze(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio)
    midi_path = Path(args.midi)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = _audio_activity(audio_path)
    audio = activity.pop("audio")
    original_midi = pretty_midi.PrettyMIDI(str(midi_path))
    first_note = _first_note_seconds(original_midi)
    active_start = activity["activeStartSeconds"]

    if active_start is None or first_note is None:
        gap = None
        retry = False
        reason = "No se pudo medir el inicio activo o el MIDI no contiene notas."
    else:
        gap = max(0.0, first_note - active_start)
        retry = active_start <= args.active_start_limit and gap >= args.gap_limit
        reason = (
            "Se detectó audio desde el comienzo, pero la primera nota MIDI "
            "aparece tarde; se ejecutará una pasada con preámbulo."
            if retry else
            "El inicio del MIDI es compatible con la actividad detectada; "
            "no se necesita rescate de borde."
        )

    padded_path = output_dir / "grabacion_con_preroll.wav"
    if retry:
        prefix_samples = int(round(args.preroll_seconds * activity["sampleRate"]))
        prefix = np.zeros(
            (prefix_samples, audio.shape[1]), dtype=np.float32
        )
        sf.write(
            str(padded_path),
            np.concatenate((prefix, audio), axis=0),
            activity["sampleRate"],
            subtype="PCM_16",
        )

    report = {
        "version": "1.0",
        "status": "retry_requested" if retry else "checked",
        **activity,
        "firstMagentaNoteSeconds": first_note,
        "gapFromDetectedAudioStartSeconds": gap,
        "prerollSeconds": args.preroll_seconds if retry else 0.0,
        "paddedAudioPath": str(padded_path) if retry else None,
        "shouldRetry": retry,
        "reason": reason,
        "automaticDeletion": False,
    }
    report_path = output_dir / "start_boundary_diagnostic.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BOUNDARY_ANALYSIS:" + json.dumps(report, ensure_ascii=False))


def _find_instrument(
    midi: pretty_midi.PrettyMIDI,
    padded_instrument: pretty_midi.Instrument,
) -> pretty_midi.Instrument:
    for instrument in midi.instruments:
        if (
            not instrument.is_drum
            and instrument.program == padded_instrument.program
        ):
            return instrument

    instrument = pretty_midi.Instrument(
        program=padded_instrument.program,
        is_drum=False,
        name=padded_instrument.name or "Notas recuperadas al inicio",
    )
    midi.instruments.append(instrument)
    return instrument


def merge_candidates(args: argparse.Namespace) -> None:
    original_path = Path(args.original_midi)
    padded_path = Path(args.padded_midi)
    output_path = Path(args.output_midi)
    output_dir = Path(args.output_dir)

    original = pretty_midi.PrettyMIDI(str(original_path))
    padded = pretty_midi.PrettyMIDI(str(padded_path))
    first_original = _first_note_seconds(original)
    if first_original is None:
        print(json.dumps({"addedNotes": 0, "reason": "El MIDI original no contiene notas."}))
        return

    activity = _audio_activity(Path(args.audio))
    active_start = activity["activeStartSeconds"]
    if active_start is None:
        print(json.dumps({"addedNotes": 0, "reason": "No se detectó actividad de audio."}))
        return

    candidates = []
    accepted_keys = []
    duplicate_tolerance = args.duplicate_tolerance
    # Incluye notas del mismo acorde inicial, no solo eventos estrictamente
    # anteriores al primer pitch que Magenta alcanzó a detectar.
    latest_candidate_start = first_original + args.guard_seconds

    original_notes = [
        note
        for instrument in original.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]

    for padded_instrument in padded.instruments:
        if padded_instrument.is_drum:
            continue
        for note in padded_instrument.notes:
            start = float(note.start) - args.preroll_seconds
            end = float(note.end) - args.preroll_seconds
            if end <= 0.0:
                continue
            start = max(0.0, start)
            end = max(start + 0.025, end)
            if start < active_start - args.active_guard_seconds:
                continue
            if start >= latest_candidate_start:
                continue
            if end - start < args.minimum_duration_seconds:
                continue

            duplicate = any(
                int(existing.pitch) == int(note.pitch)
                and abs(float(existing.start) - start) <= duplicate_tolerance
                for existing in original_notes
            )
            duplicate = duplicate or any(
                pitch == int(note.pitch)
                and abs(existing_start - start) <= duplicate_tolerance
                for pitch, existing_start in accepted_keys
            )
            if duplicate:
                continue

            candidates.append({
                "pitch": int(note.pitch),
                "pitchName": pretty_midi.note_number_to_name(int(note.pitch)),
                "startSeconds": start,
                "endSeconds": end,
                "velocity": int(note.velocity),
                "program": int(padded_instrument.program),
            })
            accepted_keys.append((int(note.pitch), start))

    candidates.sort(key=lambda item: (item["startSeconds"], item["pitch"]))
    corrected = copy.deepcopy(original)
    for candidate in candidates:
        instrument = next(
            (
                item for item in corrected.instruments
                if not item.is_drum and item.program == candidate["program"]
            ),
            None,
        )
        if instrument is None:
            instrument = pretty_midi.Instrument(
                program=candidate["program"],
                name="Notas recuperadas al inicio",
            )
            corrected.instruments.append(instrument)
        instrument.notes.append(pretty_midi.Note(
            velocity=candidate["velocity"],
            pitch=candidate["pitch"],
            start=candidate["startSeconds"],
            end=candidate["endSeconds"],
        ))

    for instrument in corrected.instruments:
        instrument.notes.sort(key=lambda note: (note.start, note.pitch))

    if candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.write(str(output_path))

    report = {
        "version": "1.0",
        "status": "candidates_found" if candidates else "no_candidates",
        "originalMidiPath": str(original_path),
        "paddedMidiPath": str(padded_path),
        "correctedCandidateMidiPath": str(output_path) if candidates else None,
        "activeAudioStartSeconds": active_start,
        "firstOriginalNoteSeconds": first_original,
        "prerollSecondsRemoved": args.preroll_seconds,
        "candidateCount": len(candidates),
        "candidates": candidates,
        "candidatePolicy": (
            "Solo se añaden a este MIDI candidato las notas nuevas detectadas "
            "en la pasada con preámbulo antes o junto al primer ataque original. "
            "El MIDI principal original se conserva intacto."
        ),
        "automaticDeletion": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "start_boundary_recovery.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BOUNDARY_RECOVERY:" + json.dumps(report, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--audio", required=True)
    analyze_parser.add_argument("--midi", required=True)
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--preroll-seconds", type=float, default=1.0)
    analyze_parser.add_argument("--active-start-limit", type=float, default=0.15)
    analyze_parser.add_argument("--gap-limit", type=float, default=0.45)
    analyze_parser.set_defaults(handler=analyze)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--audio", required=True)
    merge_parser.add_argument("--original-midi", required=True)
    merge_parser.add_argument("--padded-midi", required=True)
    merge_parser.add_argument("--output-midi", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--preroll-seconds", type=float, default=1.0)
    merge_parser.add_argument("--duplicate-tolerance", type=float, default=0.10)
    merge_parser.add_argument("--guard-seconds", type=float, default=0.06)
    merge_parser.add_argument("--active-guard-seconds", type=float, default=0.08)
    merge_parser.add_argument("--minimum-duration-seconds", type=float, default=0.045)
    merge_parser.set_defaults(handler=merge_candidates)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
