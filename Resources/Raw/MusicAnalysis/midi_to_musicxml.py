"""Exporta el MIDI de piano como un gran pentagrama de Sol y Fa."""
import copy
import json
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
ONSET_GROUP_TOLERANCE_QL = 0.06
# Incluye subdivisiones binarias y ternarias. La penalización por complejidad
# conserva la notación más sencilla que explique los ataques.
RHYTHMIC_SUBDIVISIONS = (1, 2, 3, 4, 6, 8)
MAX_INITIAL_ONSET_COMPENSATION_QL = 0.125
TIE_POSITION_TOLERANCE_QL = 1e-6


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
    """Extrae notas y recompone las que music21 dividió con ligaduras."""
    source_parts = list(source_score.parts)
    if not source_parts:
        source_parts = [source_score]

    records = []
    for part_index, source_part in enumerate(source_parts):
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
                tie = getattr(source_note, "tie", None)
                if tie is None:
                    tie = getattr(event, "tie", None)
                records.append({
                    "pitch": int(source_note.pitch.midi),
                    "pitchObject": copy.deepcopy(source_note.pitch),
                    "startQL": event_offset,
                    "durationQL": event_duration,
                    "velocity": int(velocity),
                    "_tieType": getattr(tie, "type", None),
                    "_sourcePartIndex": part_index,
                })

    records.sort(key=lambda item: (
        item["startQL"],
        item["pitch"],
        item["_sourcePartIndex"],
    ))

    merged_records = []
    active_ties = {}
    merged_fragment_count = 0

    for record in records:
        tie_type = record.pop("_tieType", None)
        part_index = record.pop("_sourcePartIndex")
        tie_key = (part_index, record["pitch"])
        candidates = active_ties.get(tie_key, [])
        previous = None

        if tie_type in ("continue", "stop"):
            matching = [
                candidate
                for candidate in candidates
                if abs(
                    candidate["startQL"] + candidate["durationQL"] -
                    record["startQL"]
                ) <= TIE_POSITION_TOLERANCE_QL
            ]
            if matching:
                previous = matching[-1]

        if previous is not None:
            segment_end = record["startQL"] + record["durationQL"]
            previous["durationQL"] = segment_end - previous["startQL"]
            merged_fragment_count += 1

            if tie_type == "stop":
                active_ties[tie_key] = [
                    candidate
                    for candidate in candidates
                    if candidate is not previous
                ]
            continue

        merged_records.append(record)
        if tie_type in ("start", "continue"):
            active_ties.setdefault(tie_key, []).append(record)

    return merged_records, len(records), merged_fragment_count


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _group_onsets(records, tolerance=None):
    """Agrupa ataques casi simultáneos para absorber el jitter del transcriptor."""
    if tolerance is None:
        tolerance = ONSET_GROUP_TOLERANCE_QL

    groups = []
    for record in sorted(records, key=lambda item: item["startQL"]):
        if (
            not groups
            or record["startQL"] - groups[-1][0]["startQL"]
            > tolerance + 1e-9
        ):
            groups.append([])
        groups[-1].append(record)
    return groups


def estimate_rhythmic_grid(records):
    """Elige la cuadrícula más sencilla que explica casi todos los ataques.

    Los ataques casi simultáneos se miden como un solo grupo: el retraso de una
    nota dentro de un acorde no debe obligar a usar una cuadrícula más fina.
    Un pequeño desfase inicial se trata como latencia de transcripción. Los
    ataques fuera de cuadrícula cuentan en el error para que una mediana sola
    no los oculte.
    """
    onsets = [
        statistics.median(record["startQL"] for record in group)
        for group in _group_onsets(records)
    ]
    if len(onsets) < 3:
        return 4, 0.0, 0.0, 0.0

    # Magenta puede situar el primer ataque unos cuadros después del pulso.
    # Solo se compensa un desfase pequeño cerca de cero; un retraso mayor puede
    # ser una anacrusa o un silencio inicial real y no debe eliminarse.
    first_onset = onsets[0]
    origin = (
        first_onset
        if 0.0 < first_onset <= MAX_INITIAL_ONSET_COMPENSATION_QL
        else 0.0
    )
    relative_onsets = [onset - origin for onset in onsets]

    candidates = []
    for subdivisions in RHYTHMIC_SUBDIVISIONS:
        step = 1.0 / subdivisions
        errors = [
            abs(onset - round(onset / step) * step)
            for onset in relative_onsets
        ]
        mean_error = statistics.mean(errors)
        upper_error = _percentile(errors, 0.95)

        # El percentil 95 considera las síncopas minoritarias; la penalización
        # sigue prefiriendo una escritura sencilla cuando el ajuste es similar.
        complexity_cost = 0.012 * math.log2(subdivisions)
        score = mean_error + 0.35 * upper_error + complexity_cost
        candidates.append((score, subdivisions, mean_error, upper_error))

    _, subdivisions, mean_error, upper_error = min(candidates)
    return subdivisions, mean_error, upper_error, origin


def quantize_records(records):
    """Alinea grupos de ataques y duraciones a una cuadrícula común."""
    (
        subdivisions,
        mean_error,
        upper_error,
        origin,
    ) = estimate_rhythmic_grid(records)
    step = 1.0 / subdivisions

    for group in _group_onsets(records):
        representative_onset = statistics.median(
            record["startQL"] for record in group
        )
        quantized_start = round(
            (representative_onset - origin) / step
        ) * step
        for record in group:
            quantized_duration = round(record["durationQL"] / step) * step
            if quantized_duration <= 0:
                quantized_duration = step
            record["startQL"] = quantized_start
            record["durationQL"] = quantized_duration

    records.sort(key=lambda item: (item["startQL"], item["pitch"]))
    return subdivisions, mean_error, upper_error, origin


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


def _infer_group_hand_boundary(group, split_point, low_center, high_center):
    """Busca una separación amplia entre bajo y acorde/melodía simultáneos."""
    pitches = sorted({record["pitch"] for record in group})
    if len(pitches) < 2:
        return None

    candidates = []
    for index in range(1, len(pitches)):
        gap = pitches[index] - pitches[index - 1]
        if gap < 12:
            continue

        low_median = statistics.median(pitches[:index])
        high_median = statistics.median(pitches[index:])
        if low_median > split_point - 8 or high_median < split_point - 8:
            continue

        direct_cost = abs(low_median - low_center) + abs(high_median - high_center)
        reversed_cost = abs(low_median - high_center) + abs(high_median - low_center)
        confidence = reversed_cost - direct_cost
        if confidence < 5:
            continue

        # Favorece un hueco amplio y una división coherente con los centros
        # de registro observados a lo largo de toda la pieza.
        candidates.append((gap + confidence * 0.35, pitches[index]))

    if not candidates:
        return None
    return max(candidates)[1]


def _find_repeated_waltz_accompaniment_groups(records, measure_length):
    """Detecta acordes repetidos tras el bajo inicial de un vals en 3/4."""
    if (
        os.environ.get("METER_NUMERATOR") != "3"
        or os.environ.get("METER_DENOMINATOR") != "4"
    ):
        return set()

    groups = _group_onsets(records)
    bass_downbeat_measures = set()
    for group in groups:
        group_start = statistics.median(
            record["startQL"] for record in group
        )
        measure_index = int(
            math.floor((group_start + 1e-7) / measure_length)
        )
        measure_offset = group_start - measure_index * measure_length
        if (
            measure_offset <= 0.25 + 1e-7
            and any(record["pitch"] <= 48 for record in group)
        ):
            bass_downbeat_measures.add(measure_index)

    candidates = []
    measures_by_slot = {}
    for group in groups:
        group_start = statistics.median(
            record["startQL"] for record in group
        )
        measure_index = int(
            math.floor((group_start + 1e-7) / measure_length)
        )
        if measure_index not in bass_downbeat_measures:
            continue

        measure_offset = group_start - measure_index * measure_length
        if measure_offset <= 0.25 + 1e-7:
            continue
        # El MIDI ya está cuantizado como máximo a semicorcheas. Reconoce
        # posiciones repetidas aunque el acorde caiga entre pulsos principales.
        slot = round(measure_offset * 4) / 4.0

        chord_pitches = sorted({
            record["pitch"]
            for record in group
            if 55 <= record["pitch"] <= 65
        })
        if (
            len(chord_pitches) < 2
            or chord_pitches[-1] - chord_pitches[0] < 5
            or statistics.median(chord_pitches) < 58
        ):
            continue

        candidates.append((id(group[0]), slot, measure_index))
        measures_by_slot.setdefault(slot, set()).add(measure_index)

    # Un patrón de acompañamiento se acepta solo si reaparece en al menos tres
    # compases con bajo en el primer pulso; no depende de un único ritmo fijo.
    repeated_slots = {
        slot
        for slot, measure_indices in measures_by_slot.items()
        if len(measure_indices) >= 3
    }

    return {
        group_id
        for group_id, slot, _ in candidates
        if slot in repeated_slots
    }


def assign_hands(records, split_point, low_center, high_center):
    """Asigna el bajo y reconoce acompañamientos de vals repetidos."""
    centers = (low_center, high_center)
    hands = {"left": [], "right": []}
    previous_pitch = {"left": None, "right": None}
    separated_groups = 0
    midrange_chords_to_treble = 0
    waltz_groups_to_bass = 0
    waltz_tones_to_bass = 0
    measure_length = _measure_length_from_environment() or 4.0
    waltz_accompaniment_groups = _find_repeated_waltz_accompaniment_groups(
        records,
        measure_length,
    )
    bass_measures = {
        int(math.floor((record["startQL"] + 1e-7) / measure_length))
        for record in records
        if record["pitch"] <= 48
    }

    for group in _group_onsets(records):
        group_start = group[0]["startQL"]
        assignments = []
        group_load = Counter()
        measure_index = int(
            math.floor((group_start + 1e-7) / measure_length)
        )
        midrange_pitches = sorted({
            record["pitch"]
            for record in group
            if 55 <= record["pitch"] <= 72
        })
        is_supported_midrange_chord = (
            measure_index in bass_measures
            and len(midrange_pitches) >= 2
            and midrange_pitches[-1] - midrange_pitches[0] >= 5
            and statistics.median(midrange_pitches) >= 58
        )
        group_boundary = _infer_group_hand_boundary(
            group,
            split_point,
            low_center,
            high_center,
        )
        assigned_ids = set()
        is_waltz_accompaniment = (
            id(group[0]) in waltz_accompaniment_groups
            and is_supported_midrange_chord
        )
        if is_waltz_accompaniment:
            # El acompañamiento reconocido tiene prioridad sobre la división
            # genérica por salto: esa división separaba el bajo y dejaba el
            # acorde completo en el pentagrama de Sol.
            waltz_groups_to_bass += 1
            for record in group:
                hand = "left" if record["pitch"] <= 65 else "right"
                assignments.append((record, hand))
                assigned_ids.add(id(record))
                group_load[hand] += 1
                if hand == "left" and record["pitch"] >= 55:
                    waltz_tones_to_bass += 1
        elif group_boundary is not None:
            separated_groups += 1
            for record in group:
                hand = "left" if record["pitch"] < group_boundary else "right"
                assignments.append((record, hand))
                assigned_ids.add(id(record))
                group_load[hand] += 1
        else:
            if is_supported_midrange_chord:
                midrange_chords_to_treble += 1
                for record in group:
                    if 55 <= record["pitch"] <= 72:
                        assignments.append((record, "right"))
                        assigned_ids.add(id(record))
                        group_load["right"] += 1

        # Fija los registros claros; reserva solo la zona media para continuidad.
        ambiguous = []
        for record in group:
            if id(record) in assigned_ids:
                continue
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
            hand = "right" if right_cost <= left_cost else "left"
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
                assigned_pitches.sort()
                previous_pitch[hand] = assigned_pitches[
                    len(assigned_pitches) // 2
                ]

    print("GRUPOS SEPARADOS POR SALTO DE ALTURA:", separated_groups)
    print("ACORDES REPETIDOS DE VALS EN FA:", waltz_groups_to_bass, "notas:", waltz_tones_to_bass)
    print("ACORDES MEDIOS ASIGNADOS A SOL POR PATRÓN DE BAJO:", midrange_chords_to_treble)
    return hands


def infer_notated_durations(hands, subdivisions):
    """Regulariza duraciones por mano según el espacio hasta el siguiente ataque."""
    step = 1.0 / subdivisions
    adjusted_events = 0

    for records in hands.values():
        groups = _group_onsets(records, tolerance=1e-6)
        for index, group in enumerate(groups):
            raw_duration = statistics.median(
                record["durationQL"] for record in group
            )
            if index + 1 < len(groups):
                next_onset = statistics.median(
                    record["startQL"] for record in groups[index + 1]
                )
                gap = max(step, next_onset - group[0]["startQL"])
                target_duration = min(gap, max(raw_duration, gap * 0.5))
                target_duration = round(target_duration / step) * step
                target_duration = min(gap, max(step, target_duration))
            else:
                target_duration = max(
                    step,
                    round(raw_duration / step) * step,
                )

            for record in group:
                if abs(record["durationQL"] - target_duration) > 1e-6:
                    adjusted_events += 1
                record["durationQL"] = target_duration

    print("DURACIONES REGULARIZADAS POR ESPACIADO:", adjusted_events)
    return adjusted_events


def _unit_quarter_length(tempo_unit):
    unit_quarter_lengths = {
        "eighth": 0.5,
        "quarter": 1.0,
        "dotted_quarter": 1.5,
        "half": 2.0,
    }
    normalized_unit = (tempo_unit or "").strip().lower()
    if normalized_unit not in unit_quarter_lengths:
        raise ValueError(f"Unidad de tempo no soportada: {tempo_unit}")
    return unit_quarter_lengths[normalized_unit]


def _quarter_bpm_for_mark(mark):
    """Obtiene negras por minuto a partir de una marca de tempo."""
    try:
        quarter_bpm = float(mark.getQuarterBPM())
        if math.isfinite(quarter_bpm) and quarter_bpm > 0:
            return quarter_bpm
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass

    try:
        number = float(mark.number)
        referent = getattr(mark, "referent", None)
        referent_length = (
            float(referent.quarterLength) if referent is not None else 1.0
        )
        quarter_bpm = number / referent_length
        if math.isfinite(quarter_bpm) and quarter_bpm > 0:
            return quarter_bpm
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def source_tempo_quarter_bpm(source_score):
    """Lee el tempo constante del MIDI; los MIDI de los motores usan 120."""
    try:
        flat_score = source_score.flatten()
        marks = list(flat_score.getElementsByClass(tempo.MetronomeMark))
    except (AttributeError, TypeError):
        marks = []

    tempo_points = {}
    for mark in marks:
        quarter_bpm = _quarter_bpm_for_mark(mark)
        if quarter_bpm is None:
            continue
        try:
            offset = float(mark.offset)
        except (AttributeError, TypeError, ValueError):
            offset = 0.0
        tempo_points[round(offset, 6)] = quarter_bpm

    if not tempo_points:
        return 120.0, False, "predeterminado 120"

    distinct_tempos = {round(value, 6) for value in tempo_points.values()}
    if len(distinct_tempos) > 1:
        return None, True, "MIDI con cambios de tempo"

    return next(iter(tempo_points.values())), False, "MIDI"


def normalize_records_to_tempo(records, source_score):
    """Convierte la cuadrícula MIDI al tempo solicitado preservando segundos."""
    tempo_value = os.environ.get("TEMPO_BPM")
    tempo_unit = os.environ.get("TEMPO_UNIT")
    if not tempo_value or not tempo_unit:
        print("ESCALA TEMPORAL: sin cambio; tempo objetivo no configurado")
        return 1.0

    try:
        bpm = float(tempo_value)
        referent_length = _unit_quarter_length(tempo_unit)
    except (TypeError, ValueError) as exc:
        print("ESCALA TEMPORAL: sin cambio; configuración no válida", str(exc))
        return 1.0

    if not math.isfinite(bpm) or bpm <= 0:
        print("ESCALA TEMPORAL: sin cambio; BPM no válido")
        return 1.0

    source_qpm, has_tempo_changes, source_label = (
        source_tempo_quarter_bpm(source_score)
    )
    if has_tempo_changes:
        print(
            "ESCALA TEMPORAL: omitida; el MIDI tiene varios tempos y no se "
            "puede aplicar una escala única sin alterar sus tiempos"
        )
        return 1.0

    # TEMPO_BPM expresa cuántas veces su figura de referencia ocurre por
    # minuto. Para convertirlo a negras por minuto, multiplica por la duración
    # de esa figura en negras; dividir comprimiría la partitura para valores
    # como corchea o negra con puntillo.
    target_qpm = bpm * referent_length
    if source_qpm is None or not math.isfinite(source_qpm) or source_qpm <= 0:
        print("ESCALA TEMPORAL: omitida; no se pudo leer el tempo MIDI")
        return 1.0

    scale = target_qpm / source_qpm
    for record in records:
        record["startQL"] *= scale
        record["durationQL"] *= scale

    print(
        "ESCALA TEMPORAL:",
        f"{source_label} {source_qpm:.3f} negras/min ->",
        f"objetivo {target_qpm:.3f} negras/min; factor {scale:.6f}",
    )
    return scale


def trim_leading_full_measures(records):
    """Quita compases completos de silencio sin mover la fase del primer ataque."""
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")
    if not numerator or not denominator:
        print("SILENCIO INICIAL: se conserva; compás no configurado")
        return 0.0

    try:
        measure_length = int(numerator) * 4.0 / int(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        print("SILENCIO INICIAL: se conserva; compás no válido")
        return 0.0

    if not math.isfinite(measure_length) or measure_length <= 0:
        print("SILENCIO INICIAL: se conserva; compás no válido")
        return 0.0

    first_onset = min(record["startQL"] for record in records)
    measures_to_trim = int(math.floor((first_onset + 1e-7) / measure_length))
    if measures_to_trim <= 0:
        print("SILENCIO INICIAL: menos de un compás completo; se conserva")
        return 0.0

    amount = measures_to_trim * measure_length
    for record in records:
        record["startQL"] = max(0.0, record["startQL"] - amount)

    print(
        "SILENCIO INICIAL:",
        f"se quitaron {measures_to_trim} compases completos",
        "y se conservó la posición del ataque dentro del compás",
    )
    return amount


def align_records_to_meter_phase(records):
    """Alinea las barras con el pulso fuerte estimado en el análisis de audio.

    La fase se expresa como el tiempo del primer downbeat de la cuadrícula.
    Se escoge el downbeat más reciente que no quede después del primer ataque,
    de modo que una pausa inicial larga no se convierta en varios compases
    vacíos y que el ataque conserve su pulso dentro del compás.
    """
    phase_index = os.environ.get("METER_PHASE_INDEX")
    downbeat_seconds = os.environ.get("METER_PHASE_DOWNBEAT_SECONDS")
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")
    bpm = os.environ.get("TEMPO_BPM")
    tempo_unit = os.environ.get("TEMPO_UNIT")

    if not all((phase_index, downbeat_seconds, numerator, denominator, bpm, tempo_unit)):
        print("FASE MÉTRICA: no disponible; se conserva el origen MIDI")
        return 0.0

    try:
        phase_index_value = int(phase_index)
        downbeat_ql = float(downbeat_seconds) * float(bpm) * (
            _unit_quarter_length(tempo_unit) / 60.0
        )
        measure_length = int(numerator) * 4.0 / int(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        print("FASE MÉTRICA: configuración no válida; se conserva el origen MIDI")
        return 0.0

    if (
        phase_index_value < 0
        or not math.isfinite(downbeat_ql)
        or not math.isfinite(measure_length)
        or measure_length <= 0
    ):
        print("FASE MÉTRICA: valores no válidos; se conserva el origen MIDI")
        return 0.0

    first_onset = min(record["startQL"] for record in records)
    bar_index = math.floor((first_onset - downbeat_ql) / measure_length)
    anchor_ql = downbeat_ql + bar_index * measure_length

    for record in records:
        record["startQL"] -= anchor_ql

    print(
        "FASE MÉTRICA:",
        f"fase {phase_index_value}; downbeat de referencia",
        f"{downbeat_seconds} s; origen de compás {anchor_ql:.6f} negras",
    )
    return anchor_ql


def _insert_notatable_events(target_part, records):
    """Escribe cada ataque como nota o acorde, sin voces duplicadas."""
    for group in _group_onsets(records, tolerance=1e-6):
        start = group[0]["startQL"]
        duration_ql = statistics.median(
            record["durationQL"] for record in group
        )
        pitches = {}
        for record in group:
            pitches.setdefault(record["pitch"], record)
        unique_records = [pitches[pitch] for pitch in sorted(pitches)]

        if len(unique_records) == 1:
            event = note.Note(unique_records[0]["pitchObject"])
            event.volume.velocity = unique_records[0]["velocity"]
        else:
            event = chord.Chord([
                record["pitchObject"] for record in unique_records
            ])
            for event_note, record in zip(event.notes, unique_records):
                event_note.volume.velocity = record["velocity"]

        event.duration = duration.Duration(quarterLength=duration_ql)
        target_part.insert(start, event)


def _measure_length_from_environment():
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")
    if not numerator or not denominator:
        return None
    try:
        measure_length = int(numerator) * 4.0 / int(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return measure_length if math.isfinite(measure_length) and measure_length > 0 else None


def _merged_activity_intervals(records):
    intervals = sorted(
        (record["startQL"], record["startQL"] + record["durationQL"])
        for record in records
    )
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1e-7:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _insert_rest_gap(target_part, start, end, measure_length):
    cursor = start
    while cursor < end - 1e-7:
        next_bar = (
            math.floor((cursor + 1e-7) / measure_length) + 1
        ) * measure_length
        segment_end = min(end, next_bar)
        segment_length = segment_end - cursor
        starts_at_bar = (
            abs(cursor - round(cursor / measure_length) * measure_length) < 1e-6
        )
        is_full_measure = (
            starts_at_bar
            and abs(segment_length - measure_length) < 1e-6
        )
        if not is_full_measure:
            rest = note.Rest()
            rest.duration = duration.Duration(quarterLength=segment_length)
            target_part.insert(cursor, rest)
        cursor = segment_end


def fill_silent_gaps(target_part, records, total_end, measure_length):
    """Inserta silencios simples en huecos parciales y conserva compases vacíos."""
    intervals = _merged_activity_intervals(records)
    cursor = 0.0
    for start, end in intervals:
        if start > cursor + 1e-7:
            _insert_rest_gap(target_part, cursor, start, measure_length)
        cursor = max(cursor, end)
    if total_end > cursor + 1e-7:
        _insert_rest_gap(target_part, cursor, total_end, measure_length)


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

    _insert_notatable_events(right_staff, hands["right"])
    _insert_notatable_events(left_staff, hands["left"])

    measure_length = _measure_length_from_environment()
    if measure_length is not None:
        all_records = hands["right"] + hands["left"]
        final_offset = max(
            record["startQL"] + record["durationQL"]
            for record in all_records
        )
        measure_count = max(
            1,
            int(math.ceil((final_offset - 1e-7) / measure_length)),
        )
        total_end = measure_count * measure_length
        fill_silent_gaps(right_staff, hands["right"], total_end, measure_length)
        fill_silent_gaps(left_staff, hands["left"], total_end, measure_length)

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
    referent_length = _unit_quarter_length(tempo_unit)
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

    # Conserva los tiempos originales del transcriptor. music21, de otro modo,
    # aplica su propia cuantización antes de que este script estime la cuadrícula.
    imported_score = converter.parse(MIDI_INPUT, quantizePost=False)
    records, imported_pitch_record_count, merged_tie_fragment_count = (
        extract_pitched_notes(imported_score)
    )
    if not records:
        raise ValueError("El MIDI no contiene notas con altura definida.")

    quarter_length_scale = normalize_records_to_tempo(
        records,
        imported_score,
    )
    meter_phase_anchor_ql = align_records_to_meter_phase(records)
    trimmed_quarter_lengths = trim_leading_full_measures(records)
    subdivisions, mean_error, upper_error, onset_origin_correction = (
        quantize_records(records)
    )

    split_point, low_center, high_center = estimate_hand_split(records)
    hands = assign_hands(
        records,
        split_point,
        low_center,
        high_center,
    )
    infer_notated_durations(hands, subdivisions)
    score = build_grand_staff(hands)

    print("MIDI_INPUT:", MIDI_INPUT)
    print("NOTAS IMPORTADAS ANTES DE UNIR LIGADURAS:", imported_pitch_record_count)
    print("FRAGMENTOS LIGADOS UNIDOS:", merged_tie_fragment_count)
    print("NOTAS TRAS UNIR LIGADURAS:", len(records))
    print("PUNTO DE DIVISIÓN MIDI:", split_point)
    print("CENTROS DE REGISTRO:", round(low_center, 1), round(high_center, 1))
    print(
        "CUADRÍCULA RÍTMICA:",
        f"1/{subdivisions} de negra; error medio",
        round(mean_error, 4),
        "error p95",
        round(upper_error, 4),
        "corrección de ataque inicial",
        round(onset_origin_correction, 4),
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

    timing_map_output = os.environ.get("TIMING_MAP_OUTPUT")
    if timing_map_output:
        with open(timing_map_output, "w", encoding="utf-8") as timing_map_file:
            json.dump(
                {
                    "quarterLengthScale": quarter_length_scale,
                    "meterPhaseAnchorQL": meter_phase_anchor_ql,
                    "meterPhaseIndex": (
                        int(os.environ["METER_PHASE_INDEX"])
                        if os.environ.get("METER_PHASE_INDEX")
                        else None
                    ),
                    "meterPhaseSource": os.environ.get(
                        "METER_PHASE_SOURCE"
                    ),
                    "trimmedQuarterLengths": trimmed_quarter_lengths,
                    "quantizationSubdivisions": subdivisions,
                    "onsetOriginCorrectionQL": onset_origin_correction,
                },
                timing_map_file,
                ensure_ascii=False,
            )

    print("XML_OUTPUT:", XML_OUTPUT)


if __name__ == "__main__":
    main()


