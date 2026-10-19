"""Punto de entrada del análisis de tempo y compás."""
import json
import os
import sys

from madmom.features.downbeats import RNNBarProcessor

from audio_features import build_onset_envelope, finite_array
from beat_tracking import (
    estimate_librosa,
    estimate_madmom_beats,
    build_tempo_candidates,
    describe_tempo_relation,
)
from beat_grid import grid_tempo, compare_beat_grids
from meter_scoring import estimate_meter_for_grid
from midi_diagnostics import read_midi_note_starts, evaluate_midi_alignment
from tempo_meter_selection import (
    estimate_numeric_tempo,
    select_joint_tempo_and_meter,
)

AUDIO_PATH = os.environ.get("AUDIO_INPUT")
MIDI_PATH = os.environ.get("MIDI_INPUT")

def round_times(times):
    """Redondea tiempos para incluir las cuadrículas en el JSON."""
    return [
        round(float(value), 6)
        for value in finite_array(times)
    ]

def choose_consensus_meter(
    librosa_meter,
    madmom_meter,
    grid_comparison,
):
    """
    Confirma compás solo si las cuadrículas y ambas evaluaciones coinciden.
    """
    librosa_selected = librosa_meter["selected"]
    madmom_selected = madmom_meter["selected"]

    if not librosa_selected or not madmom_selected:
        return None, None, "one_or_both_grids_are_ambiguous"

    same_meter = (
        librosa_selected["numerator"]
        == madmom_selected["numerator"]
        and librosa_selected["denominator"]
        == madmom_selected["denominator"]
    )

    if not same_meter:
        return None, None, "grid_meter_predictions_disagree"

    rate_relation = grid_comparison["rateRelation"]
    matched_fraction = grid_comparison["matchedFraction"]

    if rate_relation != "similar":
        return None, None, "pulse_rates_disagree"

    if (
        matched_fraction is None
        or matched_fraction < 0.60
    ):
        return None, None, "pulse_positions_do_not_agree_enough"

    confidence = min(
        librosa_meter["margin"],
        madmom_meter["margin"],
    )

    return librosa_selected, float(confidence), "grid_consensus"

def main():
    if not AUDIO_PATH:
        raise RuntimeError(
            "Falta la variable de entorno AUDIO_INPUT."
        )

    (
        librosa_tempo,
        librosa_beats,
        y,
        sr,
    ) = estimate_librosa(AUDIO_PATH)

    madmom_beats = estimate_madmom_beats(AUDIO_PATH)
    madmom_tempo = grid_tempo(madmom_beats)

    tempo_candidates = build_tempo_candidates(
        librosa_tempo,
        madmom_tempo,
    )

    tempo_relation = describe_tempo_relation(
        librosa_tempo,
        madmom_tempo,
    )
    grid_comparison = compare_beat_grids(
        librosa_beats,
        madmom_beats,
    )

    onset_envelope = build_onset_envelope(y, sr)

    # Se reutiliza una instancia del procesador para ambas cuadrículas.
    try:
        bar_processor = RNNBarProcessor()
    except Exception as exc:
        bar_processor = None
        print(
            "AVISO: no se pudo iniciar RNNBarProcessor: "
            f"{exc}",
            file=sys.stderr,
        )

    if bar_processor is None:
        # Las funciones aceptan el procesador; este reemplazo permitirá que
        # el cálculo basado en acentos continúe sin downbeat de Madmom.
        class EmptyBarProcessor:
            def __call__(self, _value):
                raise RuntimeError(
                    "RNNBarProcessor no está disponible."
                )

        bar_processor = EmptyBarProcessor()

    librosa_meter = estimate_meter_for_grid(
        audio_path=AUDIO_PATH,
        beat_times=librosa_beats,
        onset_envelope=onset_envelope,
        sr=sr,
        bar_processor=bar_processor,
    )
    madmom_meter = estimate_meter_for_grid(
        audio_path=AUDIO_PATH,
        beat_times=madmom_beats,
        onset_envelope=onset_envelope,
        sr=sr,
        bar_processor=bar_processor,
    )

    note_starts = read_midi_note_starts(MIDI_PATH)

    tempo_selection = select_joint_tempo_and_meter(
        tempo_candidates=tempo_candidates,
        source_grids={
            "librosa": librosa_beats,
            "madmom": madmom_beats,
        },
        note_starts=note_starts,
        onset_envelope=onset_envelope,
        audio_path=AUDIO_PATH,
        sr=sr,
        bar_processor=bar_processor,
    )

    numeric_tempo_estimate = estimate_numeric_tempo(
        librosa_tempo,
        madmom_tempo,
    )

    selected_tempo = tempo_selection["selectedTempoBpm"]
    selected_meter = tempo_selection["selectedMeter"]
    tempo_confidence = tempo_selection["tempoConfidence"]
    tempo_decision_reason = tempo_selection["tempoDecisionReason"]
    meter_confidence = tempo_selection["meterConfidence"]
    decision_reason = tempo_selection["meterDecisionReason"]
    selected_tempo_unit = tempo_selection["selectedTempoUnit"]
    selected_pulse_bpm = tempo_selection["selectedPulseBpm"]
    selected_pulse_unit = tempo_selection["selectedPulseUnit"]
    selected_grid_pulses_per_bar = (
        tempo_selection["selectedGridPulsesPerBar"]
    )

    # El compás puede confirmarse aunque el tempo siga ambiguo,
    # si ambas cuadrículas coinciden y sus análisis de compás concuerdan.
    if selected_meter is None:
        (
            consensus_meter,
            consensus_confidence,
            consensus_reason,
        ) = choose_consensus_meter(
            librosa_meter,
            madmom_meter,
            grid_comparison,
        )

        if consensus_meter is not None:
            selected_meter = consensus_meter
            meter_confidence = consensus_confidence
            decision_reason = (
                "meter_consensus_independent_of_tempo:"
                f"{consensus_reason}"
            )

            tempo_selection["selectedMeter"] = selected_meter
            tempo_selection["meterConfidence"] = meter_confidence
            tempo_selection["meterDecisionReason"] = decision_reason
            tempo_selection["selectedMeterCandidates"] = (
                librosa_meter["candidates"]
            )

    midi_alignment = {
        "librosaGrid": evaluate_midi_alignment(
            note_starts,
            librosa_beats,
        ),
        "madmomGrid": evaluate_midi_alignment(
            note_starts,
            madmom_beats,
        ),
        "note": (
            "Diagnóstico solamente; no se eliminaron ni corrigieron notas."
        ),
    }

    # Conserva la lista de candidatos de la cuadrícula principal
    # para mantener compatibilidad con el DTO actual de C#.
    primary_meter_candidates = librosa_meter["candidates"]

    if not primary_meter_candidates:
        primary_meter_candidates = madmom_meter["candidates"]

    if tempo_selection["selectedMeterCandidates"]:
        primary_meter_candidates = tempo_selection[
            "selectedMeterCandidates"
        ]

    result = {
        # Estimaciones originales de los rastreadores.
        "tempoBpm": librosa_tempo,
        "madmomTempoBpm": madmom_tempo,
        "tempoCandidatesBpm": tempo_candidates,

        # Tempo seleccionado y significado de la cuadrícula que lo respalda.
        "selectedTempoBpm": selected_tempo,
        "selectedTempoUnit": selected_tempo_unit,
        "selectedPulseBpm": selected_pulse_bpm,
        "selectedPulseUnit": selected_pulse_unit,
        "selectedGridPulsesPerBar": selected_grid_pulses_per_bar,
        "tempoConfidence": tempo_confidence,
        "tempoDecisionReason": tempo_decision_reason,
        "tempoHypotheses": tempo_selection["tempoHypotheses"],

        "estimatedTempoBpm":
            numeric_tempo_estimate["estimatedTempoBpm"],
        "tempoEstimateSource":
            numeric_tempo_estimate["tempoEstimateSource"],
        "tempoEstimatorAgreement":
            numeric_tempo_estimate["tempoEstimatorAgreement"],

        "beatCount": int(len(madmom_beats)),

        # El compás se representa en los campos que consume C#.
        "timeSignatureNumerator": (
            selected_meter["numerator"]
            if selected_meter
            else None
        ),
        "timeSignatureDenominator": (
            selected_meter["denominator"]
            if selected_meter
            else None
        ),
        "meterConfidence": meter_confidence,
        "meterCandidates": primary_meter_candidates,

        # Datos de diagnóstico adicionales.
        "tempoRelation": tempo_relation,
        "beatGridComparison": grid_comparison,
        "meterCandidatesByGrid": {
            "librosa": librosa_meter,
            "madmom": madmom_meter,
        },
        "meterDecisionReason": decision_reason,
        "librosaBeatTimes": round_times(librosa_beats),
        "madmomBeatTimes": round_times(madmom_beats),
        "midiGridAlignment": midi_alignment,
        }

    print("TEMPO LIBROSA:", librosa_tempo)
    print("TEMPO MADMOM:", madmom_tempo)
    print("CANDIDATOS TEMPO:", tempo_candidates)
    print("TEMPO SELECCIONADO:", selected_tempo)
    print("MOTIVO SELECCIÓN TEMPO:", tempo_decision_reason)
    print(
        "HIPÓTESIS TEMPO/COMPÁS:",
        json.dumps(
            tempo_selection["tempoHypotheses"][:8],
            allow_nan=False,
        ),
    )
    print(
        "RELACIÓN DE TEMPOS:",
        json.dumps(tempo_relation, allow_nan=False),
    )
    print(
        "COMPARACIÓN DE CUADRÍCULAS:",
        json.dumps(grid_comparison, allow_nan=False),
    )
    print(
        "COMPASES EN CUADRÍCULA LIBROSA:",
        json.dumps(librosa_meter, allow_nan=False),
    )
    print(
        "COMPASES EN CUADRÍCULA MADMOM:",
        json.dumps(madmom_meter, allow_nan=False),
    )
    print(
        "ALINEACIÓN MIDI:",
        json.dumps(midi_alignment, allow_nan=False),
    )

    if selected_meter is None:
        print(
            "COMPÁS CONFIRMADO: ambiguo;",
            decision_reason,
        )
    else:
        print(
            "COMPÁS CONFIRMADO:",
            f"{selected_meter['numerator']}/"
            f"{selected_meter['denominator']}",
        )

    print(
        "TEMPO NUMÉRICO INDEPENDIENTE:",
        json.dumps(numeric_tempo_estimate, allow_nan=False),
    )

    print(
        "ANALYSIS_JSON:"
        + json.dumps(result, allow_nan=False)
    )


if __name__ == "__main__":
    main()
