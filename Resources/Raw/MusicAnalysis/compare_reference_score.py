#!/usr/bin/env python3
"""Compare Magenta/Basic Pitch output with a MusicXML or MXL piano score.

This is a review tool: it does not alter the original transcription. It reads
the comparison JSON produced by compare_transcriptions.py, estimates how much
leading audio precedes the first score note, expands ordinary MusicXML repeats,
and checks octave disagreements against the written pitches.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
BEAT_UNIT_QUARTERS = {
    "maxima": 32.0, "long": 16.0, "breve": 8.0, "whole": 4.0,
    "half": 2.0, "quarter": 1.0, "eighth": 0.5,
    "16th": 0.25, "32nd": 0.125, "64th": 0.0625,
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if local_name(child.tag) == name]


def first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def descendants(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (item for item in element.iter() if local_name(item.tag) == name)


def child_text(element: ET.Element | None, name: str, default: str = "") -> str:
    child = first_child(element, name)
    return child.text.strip() if child is not None and child.text else default


def read_musicxml(path: Path) -> ET.Element:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            rootfile: str | None = None
            if "META-INF/container.xml" in archive.namelist():
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                for item in descendants(container, "rootfile"):
                    candidate = item.attrib.get("full-path")
                    if candidate and candidate in archive.namelist():
                        rootfile = candidate
                        break
            if rootfile is None:
                candidates = [
                    name for name in archive.namelist()
                    if name.lower().endswith((".xml", ".musicxml"))
                    and name.lower() != "meta-inf/container.xml"
                ]
                if not candidates:
                    raise ValueError("El archivo MXL no contiene un MusicXML legible.")
                rootfile = candidates[0]
            return ET.fromstring(archive.read(rootfile))
    return ET.parse(path).getroot()


def _tempo_from_direction(direction: ET.Element, current_qpm: float) -> float:
    metronomes = list(descendants(direction, "metronome"))
    if metronomes:
        metronome = metronomes[0]
        unit = child_text(metronome, "beat-unit", "quarter").lower()
        quarter_value = BEAT_UNIT_QUARTERS.get(unit, 1.0)
        dots = len(direct_children(metronome, "beat-unit-dot"))
        if dots:
            quarter_value *= 1.5 if dots == 1 else 1.75 if dots == 2 else 1.875
        try:
            per_minute = float(child_text(metronome, "per-minute"))
            if per_minute > 0:
                return per_minute * quarter_value
        except ValueError:
            pass
    for sound in descendants(direction, "sound"):
        try:
            value = float(sound.attrib.get("tempo", ""))
            if value > 0:
                return value
        except ValueError:
            continue
    return current_qpm


def _note_midi_pitch(
    pitch: ET.Element,
    key_fifths: int,
    measure_accidentals: dict[tuple[str, int], int],
) -> int:
    step = child_text(pitch, "step").upper()
    octave = int(child_text(pitch, "octave", "4"))
    alter_node = first_child(pitch, "alter")
    if alter_node is not None and alter_node.text:
        alter = int(round(float(alter_node.text)))
        measure_accidentals[(step, octave)] = alter
    elif (step, octave) in measure_accidentals:
        alter = measure_accidentals[(step, octave)]
    elif key_fifths > 0 and step in SHARP_ORDER[:key_fifths]:
        alter = 1
    elif key_fifths < 0 and step in FLAT_ORDER[: -key_fifths]:
        alter = -1
    else:
        alter = 0
    return (octave + 1) * 12 + PITCH_CLASS[step] + alter


def pitch_name(midi_pitch: int) -> str:
    return f"{PITCH_NAMES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def _read_part_names(root: ET.Element) -> dict[str, str]:
    names: dict[str, str] = {}
    for score_part in descendants(root, "score-part"):
        part_id = score_part.attrib.get("id")
        if part_id:
            names[part_id] = child_text(score_part, "part-name", part_id)
    return names


def _select_part(root: ET.Element) -> ET.Element:
    parts = direct_children(root, "part")
    if not parts:
        raise ValueError("No se encontró ninguna parte musical en el MusicXML.")
    # Piano scores usually have one part containing both staves. Select the
    # part with the largest note count when the file has multiple parts.
    return max(parts, key=lambda part: sum(1 for _ in descendants(part, "note")))


def parse_score(path: Path, bpm_override: float | None = None) -> dict[str, Any]:
    root = read_musicxml(path)
    part = _select_part(root)
    part_names = _read_part_names(root)
    part_id = part.attrib.get("id", "")
    title = child_text(root, "movement-title") or child_text(
        first_child(root, "work"), "work-title", path.stem
    )

    divisions = 1.0
    meter_numerator, meter_denominator = 4, 4
    key_fifths = 0
    qpm = 120.0
    if bpm_override is not None:
        qpm = float(bpm_override)

    measures: list[dict[str, Any]] = []
    elapsed_seconds = 0.0
    for measure_index, measure in enumerate(direct_children(part, "measure")):
        cursor = 0.0
        last_note_start = 0.0
        accidental_state: dict[tuple[str, int], int] = {}
        note_events: list[dict[str, Any]] = []
        forward_repeat = False
        backward_repeat: int | None = None

        for element in list(measure):
            tag = local_name(element.tag)
            if tag == "attributes":
                try:
                    divisions = float(child_text(element, "divisions", str(divisions)))
                except ValueError:
                    pass
                time_node = first_child(element, "time")
                if time_node is not None:
                    try:
                        meter_numerator = int(child_text(time_node, "beats", str(meter_numerator)))
                        meter_denominator = int(child_text(time_node, "beat-type", str(meter_denominator)))
                    except ValueError:
                        pass
                key_node = first_child(element, "key")
                if key_node is not None:
                    try:
                        key_fifths = int(child_text(key_node, "fifths", str(key_fifths)))
                        accidental_state.clear()
                    except ValueError:
                        pass
            elif tag == "direction":
                qpm = _tempo_from_direction(element, qpm)
            elif tag == "backup":
                cursor -= float(child_text(element, "duration", "0"))
                last_note_start = cursor
            elif tag == "forward":
                cursor += float(child_text(element, "duration", "0"))
                last_note_start = cursor
            elif tag == "barline":
                repeat = first_child(element, "repeat")
                if repeat is not None:
                    direction = repeat.attrib.get("direction", "")
                    if direction == "forward":
                        forward_repeat = True
                    elif direction == "backward":
                        try:
                            backward_repeat = max(2, int(repeat.attrib.get("times", "2")))
                        except ValueError:
                            backward_repeat = 2
            elif tag == "note":
                duration_divisions = float(child_text(element, "duration", "0"))
                is_chord = first_child(element, "chord") is not None
                start_divisions = last_note_start if is_chord else cursor
                if not is_chord:
                    last_note_start = cursor
                pitch = first_child(element, "pitch")
                is_rest = first_child(element, "rest") is not None
                tied_stop = any(
                    tie.attrib.get("type") == "stop"
                    for tie in direct_children(element, "tie")
                )
                if pitch is not None and not is_rest and not tied_stop:
                    midi_pitch = _note_midi_pitch(pitch, key_fifths, accidental_state)
                    note_events.append({
                        "pitch": midi_pitch,
                        "pitchName": pitch_name(midi_pitch),
                        "measureIndex": measure_index,
                        "sourceMeasure": measure.attrib.get("number", str(measure_index + 1)),
                        "startQuarter": start_divisions / max(divisions, 1e-9),
                        "durationQuarter": duration_divisions / max(divisions, 1e-9),
                        "voice": child_text(element, "voice"),
                        "staff": child_text(element, "staff"),
                    })
                if not is_chord:
                    cursor += duration_divisions

        measure_quarters = meter_numerator * 4.0 / max(meter_denominator, 1)
        # A local tempo direction can occur within a measure. For this review,
        # the score tempo is treated as constant over that bar.
        measure_duration_seconds = measure_quarters * 60.0 / max(qpm, 1e-9)
        for event in note_events:
            event["baseTimeSeconds"] = elapsed_seconds + event["startQuarter"] * 60.0 / max(qpm, 1e-9)
        measures.append({
            "index": measure_index,
            "number": measure.attrib.get("number", str(measure_index + 1)),
            "events": note_events,
            "quarterLength": measure_quarters,
            "durationSeconds": measure_duration_seconds,
            "tempoBpm": qpm,
            "meterNumerator": meter_numerator,
            "meterDenominator": meter_denominator,
            "forwardRepeat": forward_repeat,
            "backwardRepeat": backward_repeat,
        })
        elapsed_seconds += measure_duration_seconds

    if not measures:
        raise ValueError("La parte musical no contiene compases.")

    expanded_indices = expand_repeats(measures)
    events: list[dict[str, Any]] = []
    expanded_groups: list[dict[str, Any]] = []
    playback_time = 0.0
    for playback_index, source_index in enumerate(expanded_indices):
        measure = measures[source_index]
        for event in measure["events"]:
            copied = dict(event)
            copied["scoreTimeSeconds"] = playback_time + event["startQuarter"] * 60.0 / measure["tempoBpm"]
            copied["playbackMeasure"] = playback_index + 1
            copied["sourceMeasureIndex"] = source_index
            copied["sourceMeasure"] = measure["number"]
            copied["scoreBeat"] = 1.0 + event["startQuarter"]
            events.append(copied)
        expanded_groups.append({
            "playbackMeasure": playback_index + 1,
            "sourceMeasure": measure["number"],
            "sourceMeasureIndex": source_index,
            "startSeconds": playback_time,
            "durationSeconds": measure["durationSeconds"],
            "tempoBpm": measure["tempoBpm"],
            "meterNumerator": measure["meterNumerator"],
            "meterDenominator": measure["meterDenominator"],
        })
        playback_time += measure["durationSeconds"]

    events.sort(key=lambda item: (item["scoreTimeSeconds"], item["pitch"]))
    return {
        "title": title,
        "partName": part_names.get(part_id, ""),
        "sourceMeasureCount": len(measures),
        "playbackMeasureCount": len(expanded_indices),
        "tempoBpm": float(measures[0]["tempoBpm"]),
        "meter": f"{measures[0]['meterNumerator']}/{measures[0]['meterDenominator']}",
        "durationSeconds": playback_time,
        "events": events,
        "measures": expanded_groups,
    }


def expand_repeats(measures: list[dict[str, Any]]) -> list[int]:
    """Expand simple forward/backward repeat barlines.

    A backward repeat without an earlier forward marker uses measure 1 as its
    implicit start, matching standard MusicXML playback behavior.
    """
    expanded: list[int] = []
    repeat_start = 0
    repeat_counts: dict[tuple[int, int], int] = {}
    index = 0
    guard = 0
    while index < len(measures):
        guard += 1
        if guard > max(1000, len(measures) * 20):
            raise ValueError("La estructura de repeticiones de la partitura es circular.")
        measure = measures[index]
        if measure["forwardRepeat"]:
            repeat_start = index
        expanded.append(index)
        repeat_times = measure["backwardRepeat"]
        if repeat_times is not None:
            start = repeat_start if repeat_start <= index else 0
            key = (start, index)
            additional_passes = repeat_counts.get(key, 0)
            if additional_passes < repeat_times - 1:
                repeat_counts[key] = additional_passes + 1
                index = start
                continue
            repeat_counts.pop(key, None)
            repeat_start = index + 1
        index += 1
    return expanded


def parse_pitch_name(name: str) -> int | None:
    match = re.fullmatch(r"([A-G](?:#|b)?)(-?\d+)", (name or "").strip())
    if not match:
        return None
    pitch_class = PITCH_CLASS.get(match.group(1))
    if pitch_class is None:
        return None
    return (int(match.group(2)) + 1) * 12 + pitch_class


def group_score_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: item["scoreTimeSeconds"]):
        if not groups or abs(groups[-1]["time"] - event["scoreTimeSeconds"]) > 1e-5:
            groups.append({
                "time": event["scoreTimeSeconds"],
                "pitches": [],
                "events": [],
                "playbackMeasure": event["playbackMeasure"],
                "sourceMeasure": event["sourceMeasure"],
                "sourceMeasureIndex": event["sourceMeasureIndex"],
                "scoreBeat": event["scoreBeat"],
            })
        groups[-1]["pitches"].append(event["pitch"])
        groups[-1]["events"].append(event)
    return groups


def group_performance_events(events: list[dict[str, Any]], cluster_seconds: float) -> list[dict[str, Any]]:
    parsed = []
    for event in events:
        pitch = parse_pitch_name(str(event.get("pitchName", "")))
        try:
            start = float(event["startSeconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if pitch is not None and math.isfinite(start):
            parsed.append({"time": start, "pitch": pitch, "event": event})
    parsed.sort(key=lambda item: item["time"])
    groups: list[dict[str, Any]] = []
    for item in parsed:
        if not groups or item["time"] - groups[-1]["anchor"] > cluster_seconds:
            groups.append({"anchor": item["time"], "time": item["time"], "pitches": [], "events": []})
        groups[-1]["pitches"].append(item["pitch"])
        groups[-1]["events"].append(item["event"])
        groups[-1]["time"] = min(groups[-1]["time"], item["time"])
    return groups


def match_pitch_lists(expected: list[int], actual: list[int]) -> tuple[list[tuple[int, int, str]], list[int], list[int]]:
    remaining_actual = list(range(len(actual)))
    matches: list[tuple[int, int, str]] = []
    unmatched_expected: list[int] = []
    for expected_pitch in expected:
        exact_index = next((i for i in remaining_actual if actual[i] == expected_pitch), None)
        if exact_index is not None:
            remaining_actual.remove(exact_index)
            matches.append((expected_pitch, actual[exact_index], "exact"))
            continue
        octave_index = next((i for i in remaining_actual if abs(actual[i] - expected_pitch) == 12), None)
        if octave_index is not None:
            remaining_actual.remove(octave_index)
            matches.append((expected_pitch, actual[octave_index], "octave"))
            continue
        unmatched_expected.append(expected_pitch)
    unmatched_actual = [actual[i] for i in remaining_actual]
    return matches, unmatched_expected, unmatched_actual


def _best_group_for_target(
    target: float,
    performance_groups: list[dict[str, Any]],
    times: list[float],
    tolerance: float,
    expected: list[int] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    left = bisect.bisect_left(times, target - tolerance)
    right = bisect.bisect_right(times, target + tolerance)
    choices = performance_groups[left:right]
    if not choices:
        return None, None
    if expected is None:
        chosen = min(choices, key=lambda group: abs(group["time"] - target))
    else:
        def group_score(group: dict[str, Any]) -> tuple[float, float]:
            matches, missing, extra = match_pitch_lists(expected, group["pitches"])
            quality = sum(1.0 if kind == "exact" else 0.45 for _, _, kind in matches)
            quality -= 0.15 * len(missing) + 0.06 * len(extra)
            quality -= 0.5 * abs(group["time"] - target) / max(tolerance, 1e-6)
            return quality, -abs(group["time"] - target)
        chosen = max(choices, key=group_score)
    return chosen, chosen["time"] - target


def find_alignment(
    score_groups: list[dict[str, Any]],
    magenta_events: list[dict[str, Any]],
    max_offset: float,
    onset_tolerance: float,
    reference_tempo_bpm: float,
) -> dict[str, Any]:
    performance = group_performance_events(magenta_events, cluster_seconds=0.075)
    if not performance:
        raise ValueError("El JSON no contiene eventos Magenta utilizables.")
    performance_times = [group["time"] for group in performance]
    # Use the opening passage and its written repeat, if present. It provides
    # enough distinct pitches to identify the score-to-audio offset.
    alignment_groups = [group for group in score_groups if group["time"] <= 16.0]
    if not alignment_groups:
        alignment_groups = score_groups[: min(60, len(score_groups))]
    expected_count = sum(len(group["pitches"]) for group in alignment_groups)
    candidates: list[dict[str, Any]] = []
    offset_step = 0.01
    scale_min, scale_max, scale_step = 0.94, 1.06, 0.005
    offset_count = int(round(max_offset / offset_step)) + 1
    scale_count = int(round((scale_max - scale_min) / scale_step)) + 1
    for scale_index in range(scale_count):
        scale = scale_min + scale_index * scale_step
        for offset_index in range(offset_count):
            offset = offset_index * offset_step
            exact = octave = groups_matched = 0
            weighted_matches = 0.0
            timing_error_sum = 0.0
            timing_match_count = 0
            for reference_group in alignment_groups:
                target = offset + reference_group["time"] * scale
                detected, timing_delta = _best_group_for_target(
                    target, performance, performance_times, onset_tolerance,
                    expected=reference_group["pitches"],
                )
                if detected is None:
                    continue
                matches, _, _ = match_pitch_lists(
                    reference_group["pitches"], detected["pitches"]
                )
                exact_here = sum(kind == "exact" for _, _, kind in matches)
                octave_here = sum(kind == "octave" for _, _, kind in matches)
                exact += exact_here
                octave += octave_here
                groups_matched += bool(matches)
                if timing_delta is not None and matches:
                    timing_error_sum += abs(timing_delta)
                    timing_match_count += 1
                    # Within the onset tolerance, prefer the alignment that
                    # places attacks closest to the notated score time.
                    timing_factor = max(
                        0.0,
                        1.0 - 0.65 * abs(timing_delta) / max(onset_tolerance, 1e-6),
                    )
                    weighted_matches += (exact_here + 0.45 * octave_here) * timing_factor
            quality = weighted_matches / max(expected_count, 1)
            candidates.append({
                "offsetSeconds": round(offset, 4),
                "tempoScale": round(scale, 5),
                "estimatedTempoBpm": round(reference_tempo_bpm / scale, 3),
                "exactMatches": exact,
                "octaveEquivalentMatches": octave,
                "referenceNotesConsidered": expected_count,
                "matchedOnsetGroups": groups_matched,
                "meanTimingErrorSeconds": round(
                    timing_error_sum / timing_match_count, 5
                ) if timing_match_count else None,
                "alignmentScore": round(quality, 5),
            })
    candidates.sort(
        key=lambda item: (
            item["alignmentScore"], item["exactMatches"],
            -(item["meanTimingErrorSeconds"] or 0.0),
            -abs(item["tempoScale"] - 1.0), -item["offsetSeconds"],
        ),
        reverse=True,
    )
    best = candidates[0]
    runner_up = next((
        candidate for candidate in candidates[1:]
        if abs(candidate["offsetSeconds"] - best["offsetSeconds"]) >= 0.05
        or abs(candidate["tempoScale"] - best["tempoScale"]) >= 0.01
    ), None)
    best["runnerUpScore"] = runner_up["alignmentScore"] if runner_up else None
    best["scoreMargin"] = round(
        best["alignmentScore"] - runner_up["alignmentScore"], 5
    ) if runner_up else None
    best["confidence"] = (
        "high" if best["alignmentScore"] >= 0.70 and (best["scoreMargin"] or 0) >= 0.02
        else "medium" if best["alignmentScore"] >= 0.45
        else "low"
    )
    return best


def _read_octave_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_cleanup_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _score_group_near_audio_time(
    audio_time: float,
    score_groups: list[dict[str, Any]],
    offset: float,
    scale: float,
    tolerance: float,
) -> tuple[dict[str, Any] | None, float | None]:
    best: dict[str, Any] | None = None
    best_delta: float | None = None
    for group in score_groups:
        expected_audio_time = offset + group["time"] * scale
        delta = audio_time - expected_audio_time
        if best_delta is None or abs(delta) < abs(best_delta):
            best, best_delta = group, delta
    if best is None or best_delta is None or abs(best_delta) > tolerance:
        return None, best_delta
    return best, best_delta


def _group_event_review(
    score_groups: list[dict[str, Any]],
    performance_groups: list[dict[str, Any]],
    offset: float,
    scale: float,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    times = [group["time"] for group in performance_groups]
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for ref in score_groups:
        target = offset + ref["time"] * scale
        actual, delta = _best_group_for_target(
            target, performance_groups, times, tolerance, expected=ref["pitches"]
        )
        actual_pitches = actual["pitches"] if actual else []
        matches, missing, extra = match_pitch_lists(ref["pitches"], actual_pitches)
        exact_count = sum(kind == "exact" for _, _, kind in matches)
        octave_count = sum(kind == "octave" for _, _, kind in matches)
        status = (
            "exact_match" if exact_count == len(ref["pitches"]) and not extra
            else "partial_match" if exact_count or octave_count
            else "no_detected_onset" if actual is None
            else "pitch_mismatch"
        )
        counts[status] += 1
        rows.append({
            "scoreTimeSeconds": round(ref["time"], 4),
            "expectedAudioTimeSeconds": round(target, 4),
            "detectedAudioTimeSeconds": round(actual["time"], 4) if actual else "",
            "timingDeltaSeconds": round(delta, 4) if delta is not None else "",
            "sourceMeasure": ref["sourceMeasure"],
            "playbackMeasure": ref["playbackMeasure"],
            "scoreBeat": round(ref["scoreBeat"], 3),
            "expectedPitches": " ".join(pitch_name(p) for p in ref["pitches"]),
            "detectedPitches": " ".join(pitch_name(p) for p in actual_pitches),
            "exactMatches": exact_count,
            "octaveEquivalentMatches": octave_count,
            "missingExpectedPitches": " ".join(pitch_name(p) for p in missing),
            "extraDetectedPitches": " ".join(pitch_name(p) for p in extra),
            "status": status,
        })
    return rows, dict(counts)


def _octave_review_rows(
    rows: list[dict[str, str]],
    score_groups: list[dict[str, Any]],
    offset: float,
    scale: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    for row in rows:
        try:
            event_time = float(row.get("startSeconds", ""))
        except (ValueError, TypeError):
            continue
        magenta_pitch = parse_pitch_name(row.get("magentaPitch", ""))
        basic_pitch = parse_pitch_name(row.get("basicPitchPitch", ""))
        reference, delta = _score_group_near_audio_time(
            event_time, score_groups, offset, scale, tolerance
        )
        expected = reference["pitches"] if reference else []
        magenta_exact = magenta_pitch in expected if magenta_pitch is not None else False
        basic_exact = basic_pitch in expected if basic_pitch is not None else False
        magenta_octave = any(abs(magenta_pitch - p) == 12 for p in expected) if magenta_pitch is not None else False
        basic_octave = any(abs(basic_pitch - p) == 12 for p in expected) if basic_pitch is not None else False

        if reference is None:
            status = "no_reference_onset_nearby"
        elif magenta_exact and basic_exact:
            status = "both_pitches_written_in_score"
        elif magenta_exact and basic_octave:
            status = "magenta_matches_basic_pitch_is_octave_variant"
        elif basic_exact and magenta_octave:
            status = "basic_pitch_matches_magenta_is_octave_variant"
        elif magenta_exact:
            status = "magenta_matches_basic_pitch_not_in_score"
        elif basic_exact:
            status = "basic_pitch_matches_magenta_not_in_score"
        else:
            status = "neither_pitch_matches_reference_onset"

        reviewed.append({
            "audioStartSeconds": round(event_time, 4),
            "sourceMeasure": reference["sourceMeasure"] if reference else "",
            "playbackMeasure": reference["playbackMeasure"] if reference else "",
            "scoreBeat": round(reference["scoreBeat"], 3) if reference else "",
            "scoreTimeSeconds": round(reference["time"], 4) if reference else "",
            "timingDeltaSeconds": round(delta, 4) if delta is not None else "",
            "magentaPitch": row.get("magentaPitch", ""),
            "basicPitchPitch": row.get("basicPitchPitch", ""),
            "expectedScorePitches": " ".join(pitch_name(p) for p in expected),
            "magentaExactMatch": magenta_exact,
            "basicPitchExactMatch": basic_exact,
            "review": status,
        })
    return reviewed


def _cleanup_score_review(
    cleanup: dict[str, Any] | None,
    score_groups: list[dict[str, Any]],
    offset: float,
    scale: float,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Cross-check cleanup flags against the pitches explicitly written in score.

    This is a review aid only: a score match is positive evidence that a note is
    intentional, while a non-match is not by itself permission to delete it.
    """
    if not cleanup:
        return [], {}

    diagnostic_notes = cleanup.get("Notes", cleanup.get("notes", []))
    if not isinstance(diagnostic_notes, list):
        return [], {}

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in diagnostic_notes:
        if not isinstance(item, dict):
            continue
        note = item.get("Note", item.get("note", {}))
        if not isinstance(note, dict):
            continue

        try:
            pitch = int(note.get("Pitch", note.get("pitch")))
            start = float(note.get("StartSeconds", note.get("startSeconds")))
        except (TypeError, ValueError):
            continue

        reference, delta = _score_group_near_audio_time(
            start, score_groups, offset, scale, tolerance
        )
        expected = reference["pitches"] if reference else []
        exact = pitch in expected
        octave_equivalent = any(abs(pitch - expected_pitch) == 12 for expected_pitch in expected)
        if reference is None:
            relation = "no_score_onset_nearby"
        elif exact:
            relation = "pitch_written_in_score"
        elif octave_equivalent:
            relation = "octave_variant_of_written_pitch"
        else:
            relation = "pitch_not_written_at_aligned_onset"

        cleanup_status = str(item.get("Status", item.get("status", "")))
        octave_risk = bool(item.get("OctaveOverlapRisk", item.get("octaveOverlapRisk", False)))
        counts[relation] += 1
        if octave_risk:
            counts[f"octave_risk_{relation}"] += 1
        rows.append({
            "eventIndex": note.get("EventIndex", note.get("eventIndex", "")),
            "startSeconds": round(start, 4),
            "durationSeconds": note.get("DurationSeconds", note.get("durationSeconds", "")),
            "pitch": pitch_name(pitch),
            "cleanupStatus": cleanup_status,
            "evidenceScore": item.get("EvidenceScore", item.get("evidenceScore", "")),
            "octaveOverlapRisk": octave_risk,
            "sourceMeasure": reference["sourceMeasure"] if reference else "",
            "playbackMeasure": reference["playbackMeasure"] if reference else "",
            "scoreBeat": round(reference["scoreBeat"], 3) if reference else "",
            "scoreTimingDeltaSeconds": round(delta, 4) if delta is not None else "",
            "expectedScorePitches": " ".join(pitch_name(p) for p in expected),
            "scoreRelation": relation,
            "action": (
                "Conservar: la altura está escrita en la partitura."
                if exact else
                "Revisar con audio/partitura; no borrar automáticamente."
            ),
        })
    return rows, dict(counts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _first_boundary_info(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    score = parse_score(Path(args.score), args.bpm)
    comparison_path = Path(args.comparison_json)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    magenta = comparison.get("magentaEvents", [])
    basic_pitch = comparison.get("basicPitchEvents", [])
    if not magenta:
        raise ValueError("El JSON de comparación no contiene eventos de Magenta.")

    score_groups = group_score_events(score["events"])
    alignment = find_alignment(
        score_groups,
        magenta,
        max_offset=args.max_offset,
        onset_tolerance=args.onset_tolerance,
        reference_tempo_bpm=score["tempoBpm"],
    )
    offset = alignment["offsetSeconds"]
    scale = alignment["tempoScale"]

    magenta_groups = group_performance_events(magenta, cluster_seconds=0.075)
    magenta_review, magenta_counts = _group_event_review(
        score_groups, magenta_groups, offset, scale, args.onset_tolerance
    )
    basic_groups = group_performance_events(basic_pitch, cluster_seconds=0.075)
    basic_review, basic_counts = _group_event_review(
        score_groups, basic_groups, offset, scale, args.onset_tolerance
    )

    octave_csv_path = Path(args.octave_csv) if args.octave_csv else comparison_path.with_name("octave_discrepancy_review.csv")
    octave_rows = _read_octave_csv(octave_csv_path)
    octave_review = _octave_review_rows(
        octave_rows, score_groups, offset, scale, args.onset_tolerance
    )

    cleanup_path = Path(args.cleanup_json) if args.cleanup_json else comparison_path.with_name("midi_cleanup_diagnostic.json")
    cleanup = _read_cleanup_json(cleanup_path)
    cleanup_review, cleanup_counts = _cleanup_score_review(
        cleanup, score_groups, offset, scale, args.onset_tolerance
    )

    boundary_path = Path(args.start_boundary_json) if args.start_boundary_json else comparison_path.with_name("start_boundary_diagnostic.json")
    boundary = _first_boundary_info(boundary_path)
    first_score_group = score_groups[0] if score_groups else None
    opening_target = offset + first_score_group["time"] * scale if first_score_group else offset
    first_magenta = min(
        (float(event["startSeconds"]) for event in magenta if event.get("startSeconds") is not None),
        default=None,
    )
    opening_detected, opening_delta = _best_group_for_target(
        opening_target,
        magenta_groups,
        [group["time"] for group in magenta_groups],
        args.onset_tolerance,
        expected=first_score_group["pitches"] if first_score_group else None,
    ) if first_score_group else (None, None)
    opening_matches, opening_missing, opening_extra = match_pitch_lists(
        first_score_group["pitches"] if first_score_group else [],
        opening_detected["pitches"] if opening_detected else [],
    )
    opening_score = {
        "expectedScorePitches": [pitch_name(p) for p in first_score_group["pitches"]] if first_score_group else [],
        "scoreMeasure": first_score_group["sourceMeasure"] if first_score_group else "",
        "playbackMeasure": first_score_group["playbackMeasure"] if first_score_group else "",
        "expectedAudioTimeSeconds": round(opening_target, 4),
        "detectedAudioTimeSeconds": round(opening_detected["time"], 4) if opening_detected else None,
        "timingDeltaSeconds": round(opening_delta, 4) if opening_delta is not None else None,
        "detectedPitches": [pitch_name(p) for p in opening_detected["pitches"]] if opening_detected else [],
        "exactOrOctaveMatches": [
            {"scorePitch": pitch_name(ref), "detectedPitch": pitch_name(actual), "relation": relation}
            for ref, actual, relation in opening_matches
        ],
        "missingExpectedPitches": [pitch_name(p) for p in opening_missing],
        "extraDetectedPitches": [pitch_name(p) for p in opening_extra],
        "openingPresent": opening_detected is not None and not opening_missing,
    }

    result = {
        "version": "1.0",
        "status": "completed",
        "score": {
            "title": score["title"],
            "partName": score["partName"],
            "sourceMeasureCount": score["sourceMeasureCount"],
            "playbackMeasureCountAfterRepeats": score["playbackMeasureCount"],
            "tempoBpm": score["tempoBpm"],
            "meter": score["meter"],
            "expandedDurationSeconds": round(score["durationSeconds"], 4),
        },
        "alignment": alignment,
        "recordingStart": {
            "firstMagentaNoteSeconds": round(first_magenta, 4) if first_magenta is not None else None,
            "scoreFirstAttackExpectedAtAudioSeconds": round(opening_target, 4),
            "audioActivityStartSeconds": boundary.get("activeStartSeconds") if boundary else None,
            "openingScoreGroup": opening_score,
            "interpretation": (
                "La primera frase de la partitura aparece en el audio con el desplazamiento estimado; "
                "la actividad anterior al primer ataque de la partitura no demuestra una nota omitida."
                if opening_score["openingPresent"] and offset >= 0.15
                else "La correspondencia del primer ataque con la partitura requiere revisión manual."
            ),
            "recommendedNotationShiftSeconds": round(-offset, 4) if offset > 0.02 else 0.0,
            "automaticMidiChanged": False,
        },
        "comparisonSummary": {
            "magentaScoreEventGroups": magenta_counts,
            "basicPitchScoreEventGroups": basic_counts,
            "octaveDiscrepancyRows": len(octave_review),
            "octaveReviewCounts": dict(Counter(row["review"] for row in octave_review)),
            "cleanupScoreReviewCounts": cleanup_counts,
        },
        "octaveReviewCsv": str(Path(args.output_dir) / "reference_score_octave_review.csv"),
        "magentaScoreReviewCsv": str(Path(args.output_dir) / "reference_score_magenta_review.csv"),
        "basicPitchScoreReviewCsv": str(Path(args.output_dir) / "reference_score_basic_pitch_review.csv"),
        "cleanupScoreReviewCsv": (
            str(Path(args.output_dir) / "reference_score_cleanup_review.csv")
            if cleanup is not None else None
        ),
        "notes": [
            "El desplazamiento se estima comparando ataques y alturas de la partitura con Magenta; no se infiere del RMS del WAV.",
            "Un MIDI de salida solo se desplazaría en una etapa posterior y bajo confirmación; este análisis no modifica transcripciones.",
            "Las repeticiones estándar de MusicXML se expanden antes de comparar los eventos.",
            "Las notas marcadas por limpieza se contrastan con las alturas escritas; esta tabla informa, pero nunca elimina notas automáticamente.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_score_alignment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "reference_score_magenta_review.csv", magenta_review)
    _write_csv(output_dir / "reference_score_basic_pitch_review.csv", basic_review)
    _write_csv(output_dir / "reference_score_octave_review.csv", octave_review)
    _write_csv(output_dir / "reference_score_cleanup_review.csv", cleanup_review)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alinea una transcripción MIDI con una partitura MusicXML/MXL."
    )
    parser.add_argument("--score", required=True, help="Ruta al archivo .mxl, .musicxml o .xml de la partitura original.")
    parser.add_argument("--comparison-json", required=True, help="transcription_comparison.json generado por la app.")
    parser.add_argument("--output-dir", required=True, help="Carpeta donde escribir los informes CSV/JSON.")
    parser.add_argument("--octave-csv", help="octave_discrepancy_review.csv; por defecto se busca junto al JSON.")
    parser.add_argument("--start-boundary-json", help="Diagnóstico de inicio; por defecto se busca junto al JSON.")
    parser.add_argument("--cleanup-json", help="Diagnóstico de limpieza; por defecto se busca junto al JSON.")
    parser.add_argument("--bpm", type=float, help="Override del tempo de partitura; por defecto usa el MXL.")
    parser.add_argument("--max-offset", type=float, default=5.0, help="Máximo preámbulo que se probará en segundos (default: 5).")
    parser.add_argument("--onset-tolerance", type=float, default=0.18, help="Tolerancia para alinear ataques, en segundos.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    print("REFERENCE_SCORE_ALIGNMENT:" + json.dumps({
        "offsetSeconds": result["alignment"]["offsetSeconds"],
        "tempoScale": result["alignment"]["tempoScale"],
        "estimatedTempoBpm": result["alignment"]["estimatedTempoBpm"],
        "alignmentScore": result["alignment"]["alignmentScore"],
        "confidence": result["alignment"]["confidence"],
        "openingPresent": result["recordingStart"]["openingScoreGroup"]["openingPresent"],
        "octaveReview": result["comparisonSummary"]["octaveReviewCounts"],
        "cleanupScoreReview": result["comparisonSummary"]["cleanupScoreReviewCounts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
