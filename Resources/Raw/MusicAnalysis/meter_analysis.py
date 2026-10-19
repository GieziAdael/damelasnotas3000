"""Punto de entrada del análisis de tempo y compás."""
import json
import os
import sys

import numpy as np
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
    canonical_tempo_from_grid_hypothesis,
    estimate_numeric_tempo,
    select_joint_tempo_and_meter,
)

AUDIO_PATH = os.environ.get("AUDIO_INPUT")
MIDI_PATH = os.environ.get("MIDI_INPUT")
DBN_BAR_LENGTHS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)
MIN_DBN_BAR_CYCLE_CONSISTENCY = 0.70
MIN_DBN_SIGNATURE_SCORE_MARGIN = 0.025
MIN_DBN_LOG_LIKELIHOOD_MARGIN = 5.0
MIN_DBN_TEMPLATE_SCORE = 0.10

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


def estimate_dbn_bar_length(audio_path):
    """Estima pulsos por compás con la ruta RNN + DBN de Madmom."""
    try:
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        activations = np.asarray(
            RNNDownBeatProcessor()(audio_path),
            dtype=float,
        )

        if (
            activations.ndim != 2
            or activations.shape[1] < 2
            or activations.shape[0] < 100
            or not np.all(np.isfinite(activations[:, :2]))
        ):
            return {
                "selectedBeatsPerBar": None,
                "completedBars": 0,
                "cycleConsistency": None,
                "reason": "dbn_activation_data_is_insufficient",
            }

        processor = DBNDownBeatTrackingProcessor(
            beats_per_bar=list(DBN_BAR_LENGTHS),
            min_bpm=50,
            max_bpm=220,
            num_tempi=60,
            threshold=0,
            correct=False,
            fps=100,
        )
        # Se puntúan explícitamente los HMM de cada longitud. Madmom hace
        # esta misma comparación internamente al recibir beats_per_bar como
        # lista, pero su salida pública no identifica qué HMM ganó.
        model_results = [
            hmm.viterbi(activations)
            for hmm in processor.hmms
        ]
        model_scores = np.asarray(
            [float(result[1]) for result in model_results],
            dtype=float,
        )

        if (
            not model_scores.size
            or not np.all(np.isfinite(model_scores))
        ):
            return {
                "bestBeatsPerBar": None,
                "selectedBeatsPerBar": None,
                "completedBars": 0,
                "cycleConsistency": None,
                "modelScoreMargin": None,
                "reason": "dbn_models_returned_invalid_scores",
            }

        ranking = np.argsort(model_scores)
        best_index = int(ranking[-1])
        best_beats_per_bar = int(DBN_BAR_LENGTHS[best_index])
        model_score_margin = (
            float(model_scores[best_index] - model_scores[ranking[-2]])
            if len(ranking) > 1
            else float("inf")
        )

        best_path = model_results[best_index][0]
        state_space = (
            processor.hmms[best_index]
            .transition_model
            .state_space
        )
        beat_numbers_per_frame = (
            state_space.state_positions[best_path].astype(int) + 1
        )
        beat_indices = np.nonzero(
            np.diff(beat_numbers_per_frame)
        )[0] + 1
        beat_numbers = beat_numbers_per_frame[beat_indices]
        downbeat_indices = np.flatnonzero(beat_numbers == 1)

        if len(downbeat_indices) < 3:
            return {
                "bestBeatsPerBar": best_beats_per_bar,
                "selectedBeatsPerBar": None,
                "completedBars": max(0, len(downbeat_indices) - 1),
                "cycleConsistency": None,
                "modelScoreMargin": model_score_margin,
                "reason": "dbn_found_fewer_than_two_complete_bars",
            }

        bar_intervals = np.diff(downbeat_indices)
        consistency = float(
            np.mean(
                np.abs(bar_intervals - best_beats_per_bar) <= 1
            )
        )

        if consistency < MIN_DBN_BAR_CYCLE_CONSISTENCY:
            return {
                "bestBeatsPerBar": best_beats_per_bar,
                "selectedBeatsPerBar": None,
                "completedBars": int(len(bar_intervals)),
                "cycleConsistency": consistency,
                "modelScoreMargin": model_score_margin,
                "reason": "dbn_bar_cycles_are_not_stable",
            }

        if model_score_margin < MIN_DBN_LOG_LIKELIHOOD_MARGIN:
            return {
                "bestBeatsPerBar": best_beats_per_bar,
                "selectedBeatsPerBar": None,
                "completedBars": int(len(bar_intervals)),
                "cycleConsistency": consistency,
                "modelScoreMargin": model_score_margin,
                "reason": "dbn_bar_length_models_are_too_close",
            }

        return {
            "bestBeatsPerBar": best_beats_per_bar,
            "selectedBeatsPerBar": best_beats_per_bar,
            "completedBars": int(len(bar_intervals)),
            "cycleConsistency": consistency,
            "modelScoreMargin": model_score_margin,
            "reason": "dbn_bar_cycle_consensus",
        }

    except Exception as exc:
        print(
            "AVISO: no se pudo calcular el compás con DBN de Madmom: "
            f"{exc}",
            file=sys.stderr,
        )
        return {
            "bestBeatsPerBar": None,
            "selectedBeatsPerBar": None,
            "completedBars": 0,
            "cycleConsistency": None,
            "modelScoreMargin": None,
            "reason": "dbn_meter_unavailable",
        }


def rank_meters_for_dbn_bar_length(dbn_result, meter_results_by_grid):
    """Cruza la longitud de compás DBN con acentos de cada cuadrícula."""
    beats_per_bar = (
        dbn_result.get("selectedBeatsPerBar")
        or dbn_result.get("bestBeatsPerBar")
    )

    if beats_per_bar is None:
        return {
            "selected": None,
            "margin": None,
            "candidates": [],
            "reason": dbn_result.get(
                "reason",
                "dbn_bar_length_is_ambiguous",
            ),
        }

    scores_by_meter = {}

    for grid_name, meter_result in meter_results_by_grid.items():
        best_by_signature = {}

        for candidate in meter_result.get("pulseHypotheses", []):
            if int(candidate.get("beatsPerBar", 0)) != int(beats_per_bar):
                continue

            signature = (
                int(candidate["numerator"]),
                int(candidate["denominator"]),
            )
            current = best_by_signature.get(signature)

            if current is None or candidate["score"] > current["score"]:
                best_by_signature[signature] = candidate

        for signature, candidate in best_by_signature.items():
            scores_by_meter.setdefault(signature, []).append(
                (grid_name, candidate)
            )

    candidates = []

    for (numerator, denominator), grid_values in scores_by_meter.items():
        candidate_values = [item[1] for item in grid_values]
        selected_variant = max(
            candidate_values,
            key=lambda item: float(item["score"]),
        )
        score = float(
            np.mean([float(item["score"]) for item in candidate_values])
        )

        candidates.append({
            "numerator": numerator,
            "denominator": denominator,
            "score": score,
            "beatsPerBar": int(beats_per_bar),
            "beatUnit": selected_variant.get("beatUnit"),
            "beatUnitQuarterLength": selected_variant.get(
                "beatUnitQuarterLength"
            ),
            "supportingGrids": len(grid_values),
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if not candidates:
        return {
            "selected": None,
            "margin": None,
            "candidates": [],
            "reason": "dbn_bar_length_has_no_matching_meter_templates",
        }

    if dbn_result.get("selectedBeatsPerBar") is None:
        selected = None
        margin = (
            1.0
            if len(candidates) == 1
            else float(candidates[0]["score"] - candidates[1]["score"])
        )
    elif len(candidates) == 1:
        selected = (
            candidates[0]
            if candidates[0]["score"] >= MIN_DBN_TEMPLATE_SCORE
            else None
        )
        margin = 1.0
    else:
        margin = float(candidates[0]["score"] - candidates[1]["score"])
        selected = (
            candidates[0]
            if (
                margin >= MIN_DBN_SIGNATURE_SCORE_MARGIN
                and candidates[0]["score"] >= MIN_DBN_TEMPLATE_SCORE
            )
            else None
        )

    if selected is not None:
        reason = (
            "dbn_bar_length_and_accent_score_"
            f"{beats_per_bar}_pulses_per_bar"
        )
    elif dbn_result.get("selectedBeatsPerBar") is None:
        reason = dbn_result.get(
            "reason",
            "dbn_bar_length_is_ambiguous",
        )
    else:
        reason = (
            "dbn_bar_length_found_but_signature_evidence_is_weak"
            if candidates[0]["score"] < MIN_DBN_TEMPLATE_SCORE
            else "dbn_bar_length_found_but_signature_is_ambiguous"
        )

    return {
        "selected": selected,
        "margin": margin,
        "candidates": candidates,
        "reason": reason,
    }

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

    dbn_meter = estimate_dbn_bar_length(AUDIO_PATH)
    dbn_meter_selection = rank_meters_for_dbn_bar_length(
        dbn_meter,
        {
            "librosa": librosa_meter,
            "madmom": madmom_meter,
        },
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

    selected_meter = tempo_selection["selectedMeter"]
    meter_confidence = tempo_selection["meterConfidence"]
    decision_reason = tempo_selection["meterDecisionReason"]

    # DBN elige primero cuántos pulsos forman el ciclo del compás.
    # Los acentos de las dos cuadrículas distinguen la notación dentro de
    # ese ciclo, sin exigir que el tempo notado ya esté decidido.
    if dbn_meter_selection["selected"] is not None:
        selected_candidate = dbn_meter_selection["selected"]
        selected_meter = {
            "numerator": selected_candidate["numerator"],
            "denominator": selected_candidate["denominator"],
        }
        meter_confidence = dbn_meter_selection["margin"]
        decision_reason = dbn_meter_selection["reason"]
        tempo_selection["selectedMeter"] = selected_meter
        tempo_selection["meterConfidence"] = meter_confidence
        tempo_selection["meterDecisionReason"] = decision_reason
        tempo_selection["selectedMeterCandidates"] = (
            dbn_meter_selection["candidates"]
        )
    elif dbn_meter_selection["candidates"]:
        if selected_meter is None:
            tempo_selection["selectedMeterCandidates"] = (
                dbn_meter_selection["candidates"]
            )
            decision_reason = dbn_meter_selection["reason"]
            tempo_selection["meterDecisionReason"] = decision_reason

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
        elif selected_meter is None:
            decision_reason = (
                f"{dbn_meter_selection['reason']};"
                f"{consensus_reason}"
            )
            tempo_selection["meterDecisionReason"] = decision_reason

    selected_tempo = tempo_selection["selectedTempoBpm"]
    selected_tempo_unit = tempo_selection["selectedTempoUnit"]
    selected_pulse_bpm = tempo_selection["selectedPulseBpm"]
    selected_pulse_unit = tempo_selection["selectedPulseUnit"]
    selected_grid_pulses_per_bar = (
        tempo_selection["selectedGridPulsesPerBar"]
    )
    tempo_confidence = tempo_selection["tempoConfidence"]
    tempo_decision_reason = tempo_selection["tempoDecisionReason"]

    if (
        selected_meter is not None
        and selected_tempo is None
        and numeric_tempo_estimate["estimatedTempoBpm"] is not None
    ):
        meter_variant = next(
            (
                candidate
                for candidate in tempo_selection[
                    "selectedMeterCandidates"
                ]
                if (
                    candidate["numerator"]
                    == selected_meter["numerator"]
                    and candidate["denominator"]
                    == selected_meter["denominator"]
                )
            ),
            None,
        )

        if meter_variant is not None:
            tempo_interpretation = canonical_tempo_from_grid_hypothesis(
                numeric_tempo_estimate["estimatedTempoBpm"],
                meter_variant,
            )

            if tempo_interpretation is not None:
                selected_tempo = round(
                    tempo_interpretation["tempoBpm"],
                    1,
                )
                selected_tempo_unit = tempo_interpretation[
                    "tempoUnit"
                ]
                selected_pulse_bpm = round(
                    numeric_tempo_estimate["estimatedTempoBpm"],
                    1,
                )
                selected_pulse_unit = meter_variant["beatUnit"]
                selected_grid_pulses_per_bar = meter_variant[
                    "beatsPerBar"
                ]
                tempo_confidence = numeric_tempo_estimate[
                    "tempoEstimatorAgreement"
                ]
                tempo_decision_reason = (
                    "numeric_tempo_consensus_with_selected_meter_unit"
                )

    for hypothesis in tempo_selection["tempoHypotheses"]:
        hypothesis["isSelected"] = bool(
            selected_tempo is not None
            and selected_meter is not None
            and hypothesis["tempoUnit"] == selected_tempo_unit
            and hypothesis["numerator"] == selected_meter["numerator"]
            and hypothesis["denominator"] == selected_meter["denominator"]
            and abs(hypothesis["tempoBpm"] - selected_tempo)
                / max(selected_tempo, 1e-9)
                <= 0.02
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
        "dbnMeter": {
            **dbn_meter,
            "selectionReason": dbn_meter_selection["reason"],
            "signatureCandidates": dbn_meter_selection["candidates"],
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
        "COMPÁS DBN:",
        json.dumps(
            result["dbnMeter"],
            allow_nan=False,
        ),
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
