import copy
import os

from music21 import converter, duration, meter, tempo


MIDI_INPUT = os.environ["MIDI_INPUT"]
XML_OUTPUT = os.environ["XML_OUTPUT"]


def remove_elements_from_stream(stream, element_class):
    """Elimina elementos existentes de esa clase en el stream."""
    elements = list(
        stream.recurse().getElementsByClass(element_class)
    )

    for element in elements:
        active_site = element.activeSite

        if active_site is not None:
            active_site.remove(element)


def apply_time_signature(score):
    numerator = os.environ.get("METER_NUMERATOR")
    denominator = os.environ.get("METER_DENOMINATOR")

    # Si el análisis no confirmó un compás, deja el MIDI como está.
    if not numerator or not denominator:
        print("COMPÁS APLICADO: ninguno; análisis sin confirmar")
        return

    time_signature = meter.TimeSignature(
        f"{int(numerator)}/{int(denominator)}"
    )

    parts = list(score.parts)

    if not parts:
        remove_elements_from_stream(score, meter.TimeSignature)
        score.insert(0, time_signature)
    else:
        for part in parts:
            remove_elements_from_stream(part, meter.TimeSignature)
            part.insert(
                0,
                meter.TimeSignature(
                    f"{int(numerator)}/{int(denominator)}"
                ),
            )

    print(
        "COMPÁS APLICADO:",
        f"{numerator}/{denominator}",
    )


def apply_tempo(score):
    tempo_value = os.environ.get("TEMPO_BPM")

    if not tempo_value:
        print("TEMPO APLICADO: ninguno")
        return

    bpm = float(tempo_value)

    if bpm <= 0:
        print("TEMPO APLICADO: ninguno; valor no válido")
        return

    tempo_unit = os.environ.get(
        "TEMPO_UNIT",
        "quarter",
    ).strip().lower()

    unit_quarter_lengths = {
        "eighth": 0.5,
        "quarter": 1.0,
        "dotted_quarter": 1.5,
        "half": 2.0,
    }

    if tempo_unit not in unit_quarter_lengths:
        raise ValueError(
            f"Unidad de tempo no soportada: {tempo_unit}"
        )

    referent = duration.Duration(
        quarterLength=unit_quarter_lengths[tempo_unit]
    )

    metronome_mark = tempo.MetronomeMark(
        number=bpm,
        referent=referent,
    )

    parts = list(score.parts)

    if not parts:
        remove_elements_from_stream(
            score,
            tempo.MetronomeMark,
        )
        score.insert(0, metronome_mark)
    else:
        for part in parts:
            remove_elements_from_stream(
                part,
                tempo.MetronomeMark,
            )
            part.insert(
                0,
                tempo.MetronomeMark(
                    number=bpm,
                    referent=duration.Duration(
                        quarterLength=
                            unit_quarter_lengths[tempo_unit]
                    ),
                ),
            )

    print(
        "TEMPO APLICADO:",
        bpm,
        tempo_unit,
    )


def main():
    if not os.path.isfile(MIDI_INPUT):
        raise FileNotFoundError(
            f"No se encontró el MIDI: {MIDI_INPUT}"
        )

    score = converter.parse(MIDI_INPUT)

    print("MIDI_INPUT:", MIDI_INPUT)
    print("PARTES:", len(score.parts))

    apply_time_signature(score)
    apply_tempo(score)

    export_score = copy.deepcopy(score)

    try:
        export_score.write(
            "musicxml",
            fp=XML_OUTPUT,
        )
    except KeyError as exc:
        print(
            "AVISO: music21 falló al reorganizar los tuplets; "
            "se intentará exportar sin makeNotation.",
            str(exc),
        )

        score.write(
            "musicxml",
            fp=XML_OUTPUT,
            makeNotation=False,
        )

    print("XML_OUTPUT:", XML_OUTPUT)


if __name__ == "__main__":
    main()