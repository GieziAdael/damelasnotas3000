"""Convierte el MIDI de salida a MusicXML con duraciones notables."""
import os

from music21 import converter, duration, meter, stream, tempo

MIDI_INPUT = os.environ["MIDI_INPUT"]
XML_OUTPUT = os.environ["XML_OUTPUT"]


def remove_elements_from_stream(target_stream, element_class):
    """Elimina elementos existentes de esa clase en el stream."""
    elements = list(
        target_stream.recurse().getElementsByClass(element_class)
    )
    for element in elements:
        active_site = element.activeSite
        if active_site is not None:
            active_site.remove(element)


def split_complex_durations(source_score):
    """Reconstruye las partes dividiendo notas/rests complejos en valores notables."""
    normalized_score = stream.Score()

    for part_index, source_part in enumerate(source_score.parts):
        normalized_part = stream.Part(id=f"Part{part_index + 1}")
        normalized_part.partName = (
            getattr(source_part, "partName", None)
            or f"Instrumento {part_index + 1}"
        )

        flat_part = source_part.flatten()
        for event in flat_part.notesAndRests:
            event_offset = float(event.offset)
            pieces = event.splitAtDurations()
            cursor = event_offset

            for piece in pieces:
                # Cada pieza conserva la nota/rest original y sus ligaduras.
                normalized_part.insert(cursor, piece)
                cursor += float(piece.duration.quarterLength)

        normalized_score.insert(0, normalized_part)

    if not source_score.parts:
        # MIDI sin Parts: conserva los eventos en una parte generada.
        normalized_part = stream.Part(id="Part1")
        for event in source_score.flatten().notesAndRests:
            cursor = float(event.offset)
            for piece in event.splitAtDurations():
                normalized_part.insert(cursor, piece)
                cursor += float(piece.duration.quarterLength)
        normalized_score.insert(0, normalized_part)

    return normalized_score


def apply_time_signature(score):
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")
    if not numerator or not denominator:
        print("COMPÁS APLICADO: ninguno; análisis sin confirmar")
        return

    signature_text = f"{int(numerator)}/{int(denominator)}"
    parts = list(score.parts)
    if not parts:
        remove_elements_from_stream(score, meter.TimeSignature)
        score.insert(0, meter.TimeSignature(signature_text))
    else:
        for part in parts:
            remove_elements_from_stream(part, meter.TimeSignature)
            part.insert(0, meter.TimeSignature(signature_text))
    print("COMPÁS APLICADO:", signature_text)


def apply_tempo(score):
    tempo_value = os.environ.get("TEMPO_BPM")
    if not tempo_value:
        print("TEMPO APLICADO: ninguno")
        return

    bpm = float(tempo_value)
    if bpm <= 0:
        print("TEMPO APLICADO: ninguno; valor no válido")
        return

    tempo_unit = os.environ.get("TEMPO_UNIT", "quarter").strip().lower()
    unit_quarter_lengths = {
        "eighth": 0.5,
        "quarter": 1.0,
        "dotted_quarter": 1.5,
        "half": 2.0,
    }
    if tempo_unit not in unit_quarter_lengths:
        raise ValueError(f"Unidad de tempo no soportada: {tempo_unit}")

    referent_length = unit_quarter_lengths[tempo_unit]
    parts = list(score.parts)
    if not parts:
        remove_elements_from_stream(score, tempo.MetronomeMark)
        score.insert(0, tempo.MetronomeMark(
            number=bpm,
            referent=duration.Duration(quarterLength=referent_length),
        ))
    else:
        for part in parts:
            remove_elements_from_stream(part, tempo.MetronomeMark)
            part.insert(0, tempo.MetronomeMark(
                number=bpm,
                referent=duration.Duration(quarterLength=referent_length),
            ))
    print("TEMPO APLICADO:", bpm, tempo_unit)


def main():
    if not os.path.isfile(MIDI_INPUT):
        raise FileNotFoundError(f"No se encontró el MIDI: {MIDI_INPUT}")

    imported_score = converter.parse(MIDI_INPUT)
    score = split_complex_durations(imported_score)

    print("MIDI_INPUT:", MIDI_INPUT)
    print("PARTES:", len(score.parts))
    apply_time_signature(score)
    apply_tempo(score)

    try:
        # El stream reconstruido ya tiene duraciones separadas y puede
        # reorganizarse para crear compases normales.
        score.write("musicxml", fp=XML_OUTPUT)
    except KeyError as exc:
        print(
            "AVISO: music21 no pudo reorganizar la notación; "
            "se exportará sin crear compases automáticos.",
            str(exc),
        )
        score.write("musicxml", fp=XML_OUTPUT, makeNotation=False)

    print("XML_OUTPUT:", XML_OUTPUT)


if __name__ == "__main__":
    main()
