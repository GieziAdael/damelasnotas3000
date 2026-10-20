"""Exporta el MIDI de piano como un gran pentagrama de Sol y Fa."""
import copy
import math
import os
import statistics
from collections import Counter

from music21 import (
    chord,
    clef,
    converter,
    duration,
    instrument,
    layout,
    meter,
    note,
    stream,
    tempo,
)

MIDI_INPUT = os.environ["MIDI_INPUT"]
XML_OUTPUT = os.environ["XML_OUTPUT"]

# Rango típico donde cambia la asignación entre manos en piano.
MIN_SPLIT_POINT = 48  # C3
MAX_SPLIT_POINT = 72  # C5
DEFAULT_SPLIT_POINT = 60  # C4
AMBIGUOUS_SEMITONES = 4
ONSET_GROUP_TOLERANCE_QL = 0.035
RHYTHMIC_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16)


def weighted_mean(values):
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def estimate_hand_split(records):
    """Encuentra dos centros de registro y devuelve el umbral entre manos."""
    if len(records) < 8:
        return DEFAULT_SPLIT_POINT, 48.0, 72.0

    pitch_weights = []
    for record in records:
        # Duraciones largas aportan un poco más al registro, con límite para
        # evitar que una nota sostenida domine la partición completa.
        weight = min(2.0, max(0.5, record["durationQL"] ** 0.5))
        pitch_weights.append((record["pitch"], weight))

    pitches = sorted(value for value, _ in pitch_weights)
    low_center = float(pitches[int((len(pitches) - 1) * 0.25)])
    high_center = float(pitches[int((len(pitches) - 1) * 0.75)])

    if high_center - low_center < 7:
        return DEFAULT_SPLIT_POINT, low_center, high_center

    for _ in range(32):
        low_values = []
        high_values = []
        boundary = (low_center + high_center) / 2.0

        for pitch, weight in pitch_weights:
            target = low_values if pitch <= boundary else high_values
            target.append((pitch, weight))

        if not low_values or not high_values:
            return DEFAULT_SPLIT_POINT, low_center, high_center

        next_low = weighted_mean(low_values)
        next_high = weighted_mean(high_values)
        if abs(next_low - low_center) + abs(next_high - high_center) < 0.01:
            low_center, high_center = next_low, next_high
            break
        low_center, high_center = next_low, next_high

    if high_center - low_center < 7:
        return DEFAULT_SPLIT_POINT, low_center, high_center

    split_point = round((low_center + high_center) / 2.0)
    split_point = max(MIN_SPLIT_POINT, min(MAX_SPLIT_POINT, split_point))
    return split_point, low_center, high_center


def extract_pitched_notes(source_score):
    """Convierte notas y acordes del MIDI en eventos individuales de nota."""
    source_parts = list(source_score.parts)
    if not source_parts:
        source_parts = [source_score]

    records = []
    for source_part in source_parts:
        flat_part = source_part.flatten()
        for event in flat_part.notes:
            event_offset = float(event.offset)
            event_duration = float(event.duration.quarterLength)
            if event_duration <= 0:
                continue

            if isinstance(event, chord.Chord):
                source_notes = list(event.notes)
            elif isinstance(event, note.Note):
                source_notes = [event]
            else:
                continue

            for source_note in source_notes:
                velocity = source_note.volume.velocity
                if velocity is None:
                    velocity = 64
                records.append({
                    "pitch": int(source_note.pitch.midi),
                    "pitchObject": copy.deepcopy(source_note.pitch),
                    "startQL": event_offset,
                    "durationQL": event_duration,
                    "velocity": int(velocity),
                })

    records.sort(key=lambda item: (item["startQL"], item["pitch"]))
    return records


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def estimate_rhythmic_grid(records):
    """Elige la subdivisión más sencilla que explica los ataques del MIDI.

    Los MIDI generados por transcripción suelen contener pequeñas variaciones
    de tiempo. Sin esta etapa, music21 interpreta esas variaciones como muchas
    duraciones irregulares y las muestra como tuplets o figuras difíciles.
    """
    onsets = [record["startQL"] for record in records]
    if len(onsets) < 3:
        return 4, 0.0, 0.0

    candidates = []
    for subdivisions in RHYTHMIC_SUBDIVISIONS:
        step = 1.0 / subdivisions
        errors = [
            abs(onset - round(onset / step) * step)
            for onset in onsets
        ]
        median_error = statistics.median(errors)
        upper_error = _percentile(errors, 0.85)

        # El coste pequeño por complejidad hace que, si dos cuadrículas
        # explican igual los ataques, se prefiera la subdivisión más simple.
        complexity_cost = 0.006 * math.log2(subdivisions)
        score = median_error + 0.35 * upper_error + complexity_cost
        candidates.append((score, subdivisions, median_error, upper_error))

    _, subdivisions, median_error, upper_error = min(candidates)
    return subdivisions, median_error, upper_error


def quantize_records(records):
    """Alinea inicios y finales a una cuadrícula rítmica común."""
    subdivisions, median_error, upper_error = estimate_rhythmic_grid(records)
    step = 1.0 / subdivisions

    for record in records:
        original_start = record["startQL"]
        original_end = original_start + record["durationQL"]

        quantized_start = round(original_start / step) * step
        quantized_end = round(original_end / step) * step
        if quantized_end <= quantized_start:
            quantized_end = quantized_start + step

        record["startQL"] = quantized_start
        record["durationQL"] = quantized_end - quantized_start

    records.sort(key=lambda item: (item["startQL"], item["pitch"]))
    return subdivisions, median_error, upper_error


def _hand_cost(pitch, hand, split_point, centers, previous_pitch, group_load):
    if hand == "left":
        center = centers[0]
        cost = abs(pitch - center) * 0.35
        cost += max(0, pitch - split_point) * 1.5
    else:
        center = centers[1]
        cost = abs(pitch - center) * 0.35
        cost += max(0, split_point - pitch) * 1.5

    if previous_pitch is not None:
        leap = min(24, abs(pitch - previous_pitch))
        cost += leap * 0.45

    # Si varias notas simultáneas están cerca del límite, repartirlas evita
    # asignar automáticamente todo el acorde a la misma mano.
    cost += group_load[hand] * 1.2
    return cost


def assign_hands(records, split_point, low_center, high_center):
    """Asigna cada evento a una mano con registro y continuidad local."""
    centers = (low_center, high_center)
    hands = {"left": [], "right": []}
    previous_pitch = {"left": None, "right": None}

    cursor = 0
    while cursor < len(records):
        group_start = records[cursor]["startQL"]
        end = cursor + 1
        while (
            end < len(records)
            and records[end]["startQL"] - group_start
            <= ONSET_GROUP_TOLERANCE_QL
        ):
            end += 1

        group = records[cursor:end]
        assignments = []
        group_load = Counter()

        # Primero fija notas cuyo registro queda claramente de un lado.
        ambiguous = []
        for record in group:
            pitch = record["pitch"]
            if pitch < split_point - AMBIGUOUS_SEMITONES:
                assignments.append((record, "left"))
                group_load["left"] += 1
            elif pitch > split_point + AMBIGUOUS_SEMITONES:
                assignments.append((record, "right"))
                group_load["right"] += 1
            else:
                ambiguous.append(record)

        for record in ambiguous:
            pitch = record["pitch"]
            left_cost = _hand_cost(
                pitch,
                "left",
                split_point,
                centers,
                previous_pitch["left"],
                group_load,
            )
            right_cost = _hand_cost(
                pitch,
                "right",
                split_point,
                centers,
                previous_pitch["right"],
                group_load,
            )
            hand = "left" if left_cost <= right_cost else "right"
            assignments.append((record, hand))
            group_load[hand] += 1

        for record, hand in assignments:
            hands[hand].append(record)

        for hand in ("left", "right"):
            assigned_pitches = [
                record["pitch"]
                for record, assigned_hand in assignments
                if assigned_hand == hand
            ]
            if assigned_pitches:
                # La mediana del grupo sirve como referencia para las notas
                # fronterizas del siguiente ataque.
                assigned_pitches.sort()
                previous_pitch[hand] = assigned_pitches[
                    len(assigned_pitches) // 2
                ]

        cursor = end

    return hands


def _insert_notatable_note(target_part, record):
    """Inserta una nota y divide su duración compleja conservando ligaduras."""
    source_note = note.Note(record["pitchObject"])
    source_note.volume.velocity = record["velocity"]
    source_note.duration = duration.Duration(
        quarterLength=record["durationQL"]
    )

    cursor = record["startQL"]
    for piece in source_note.splitAtDurations():
        target_part.insert(cursor, piece)
        cursor += float(piece.duration.quarterLength)


def build_grand_staff(hands):
    """Crea dos PartStaff unidos por llave en un solo gran pentagrama."""
    score = stream.Score()
    right_staff = stream.PartStaff(id="Piano-Staff1")
    left_staff = stream.PartStaff(id="Piano-Staff2")
    right_staff.partName = "Piano"
    left_staff.partName = "Piano"

    right_staff.insert(0, instrument.Piano())
    right_staff.insert(0, clef.TrebleClef())
    left_staff.insert(0, clef.BassClef())

    for record in hands["right"]:
        _insert_notatable_note(right_staff, record)
    for record in hands["left"]:
        _insert_notatable_note(left_staff, record)

    score.insert(0, right_staff)
    score.insert(0, left_staff)
    score.insert(0, layout.StaffGroup(
        [right_staff, left_staff],
        name="Piano",
        symbol="brace",
        barTogether=True,
    ))
    return score


def remove_elements_from_stream(target_stream, element_class):
    elements = list(
        target_stream.recurse().getElementsByClass(element_class)
    )
    for element in elements:
        active_site = element.activeSite
        if active_site is not None:
            active_site.remove(element)


def apply_time_signature(score):
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")
    if not numerator or not denominator:
        print("COMPÁS APLICADO: ninguno; análisis sin confirmar")
        return

    signature_text = f"{int(numerator)}/{int(denominator)}"
    for part in score.parts:
        remove_elements_from_stream(part, meter.TimeSignature)
        part.insert(0, meter.TimeSignature(signature_text))
    print("COMPÁS APLICADO:", signature_text)


def apply_tempo(score):
    tempo_value = os.environ.get("TEMPO_BPM")
    if not tempo_value:
        print("TEMPO APLICADO: ninguno; BPM no configurado")
        return

    try:
        bpm = float(tempo_value)
    except (TypeError, ValueError):
        print("TEMPO APLICADO: ninguno; BPM no válido")
        return

    if not math.isfinite(bpm) or bpm <= 0:
        print("TEMPO APLICADO: ninguno; valor no válido")
        return

    tempo_unit_value = os.environ.get("TEMPO_UNIT")
    if not tempo_unit_value or not tempo_unit_value.strip():
        print(
            "TEMPO APLICADO: ninguno; falta TEMPO_UNIT "
            "(no se asumió negra)"
        )
        return

    tempo_unit = tempo_unit_value.strip().lower()
    unit_quarter_lengths = {
        "eighth": 0.5,
        "quarter": 1.0,
        "dotted_quarter": 1.5,
        "half": 2.0,
    }
    if tempo_unit not in unit_quarter_lengths:
        raise ValueError(f"Unidad de tempo no soportada: {tempo_unit}")

    referent_length = unit_quarter_lengths[tempo_unit]
    for index, part in enumerate(score.parts):
        remove_elements_from_stream(part, tempo.MetronomeMark)
        # Una sola indicación de tempo en el pentagrama superior evita que
        # algunos lectores dupliquen la marca en el gran pentagrama.
        if index == 0:
            part.insert(0, tempo.MetronomeMark(
                number=bpm,
                referent=duration.Duration(quarterLength=referent_length),
            ))
    print("TEMPO APLICADO:", bpm, tempo_unit)


def main():
    if not os.path.isfile(MIDI_INPUT):
        raise FileNotFoundError(f"No se encontró el MIDI: {MIDI_INPUT}")

    imported_score = converter.parse(MIDI_INPUT)
    records = extract_pitched_notes(imported_score)
    if not records:
        raise ValueError("El MIDI no contiene notas con altura definida.")

    subdivisions, median_error, upper_error = quantize_records(records)

    split_point, low_center, high_center = estimate_hand_split(records)
    hands = assign_hands(
        records,
        split_point,
        low_center,
        high_center,
    )
    score = build_grand_staff(hands)

    print("MIDI_INPUT:", MIDI_INPUT)
    print("PUNTO DE DIVISIÓN MIDI:", split_point)
    print("CENTROS DE REGISTRO:", round(low_center, 1), round(high_center, 1))
    print(
        "CUADRÍCULA RÍTMICA:",
        f"1/{subdivisions} de negra; error mediano",
        round(median_error, 4),
        "error p85",
        round(upper_error, 4),
    )
    print("NOTAS MANO DERECHA / SOL:", len(hands["right"]))
    print("NOTAS MANO IZQUIERDA / FA:", len(hands["left"]))
    print("PENTAGRAMAS:", len(score.parts))

    apply_time_signature(score)
    apply_tempo(score)

    try:
        score.write("musicxml", fp=XML_OUTPUT)
    except KeyError as exc:
        print(
            "AVISO: music21 no pudo organizar la notación; "
            "se exportará sin crear compases automáticos.",
            str(exc),
        )
        score.write("musicxml", fp=XML_OUTPUT, makeNotation=False)

    print("XML_OUTPUT:", XML_OUTPUT)


if __name__ == "__main__":
    main()


