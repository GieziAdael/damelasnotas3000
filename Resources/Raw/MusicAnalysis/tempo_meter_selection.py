"""Selección conjunta de tempo, unidad de pulso y compás."""
import numpy as np

from audio_features import HOP_LENGTH, finite_array
from beat_grid import (
    grid_tempo,
    make_subdivision_grid,
    make_grids_for_candidate_tempo,
)
from meter_scoring import estimate_meter_for_grid
from midi_diagnostics import evaluate_midi_alignment

MIN_TEMPO_HYPOTHESIS_MARGIN = 0.04
MIN_JOINT_METER_MARGIN = 0.04

def autocorrelation_score_for_tempo(
    onset_envelope,
    sr,
    tempo_bpm,
):
    """
    Puntúa la repetición de ataques de audio al periodo de un tempo.
    Regresa un valor entre 0 y 1; no interpreta por sí solo el compás.
    """
    if (
        tempo_bpm is None
        or not np.isfinite(tempo_bpm)
        or tempo_bpm <= 0
    ):
        return None

    values = finite_array(onset_envelope)

    if len(values) < 32:
        return None

    values = values - float(np.mean(values))
    nominal_lag = (
        60.0 * sr
        / (float(tempo_bpm) * HOP_LENGTH)
    )

    if nominal_lag < 3 or nominal_lag >= len(values) / 3:
        return None

    center_lag = int(round(nominal_lag))
    radius = max(1, int(round(center_lag * 0.04)))
    scores = []

    for lag in range(
        max(2, center_lag - radius),
        min(len(values) // 3, center_lag + radius) + 1,
    ):
        first = values[:-lag]
        second = values[lag:]

        denominator = (
            float(np.linalg.norm(first))
            * float(np.linalg.norm(second))
        )

        if denominator <= 1e-12:
            continue

        correlation = float(
            np.dot(first, second) / denominator
        )

        if np.isfinite(correlation):
            scores.append(
                float(np.clip(correlation, 0.0, 1.0))
            )

    return max(scores) if scores else None

def midi_alignment_score(note_starts, beat_times):
    """
    Convierte el diagnóstico de alineación MIDI en una puntuación 0..1.
    Es solo una evidencia más: notas fuera del pulso pueden ser correctas.
    """
    alignment = evaluate_midi_alignment(
        note_starts,
        beat_times,
    )

    fraction = alignment.get("fractionNearPulse")
    median_distance = alignment.get(
        "medianDistanceAsBeatFraction"
    )

    if fraction is None or median_distance is None:
        return None

    phase_score = 1.0 - min(
        1.0,
        2.0 * float(median_distance),
    )

    score = (
        0.55 * float(fraction)
        + 0.45 * phase_score
    )

    return float(np.clip(score, 0.0, 1.0))

def combine_available_scores(
    audio_score,
    midi_score,
    meter_score,
):
    """
    Combina evidencia de audio y compás.

    midi_score se conserva como diagnóstico, pero no decide el tempo:
    una cuadrícula con más pulsos puede acercar más notas por densidad.
    """
    components = []

    if audio_score is not None and np.isfinite(audio_score):
        components.append((0.50, float(audio_score)))

    if meter_score is not None and np.isfinite(meter_score):
        # Los puntajes métricos están aproximadamente en [-1, 1].
        normalized_meter = float(
            np.clip((float(meter_score) + 1.0) / 2.0, 0.0, 1.0)
        )
        components.append((0.50, normalized_meter))

    if not components:
        return None

    total_weight = sum(weight for weight, _ in components)

    if total_weight <= 0:
        return None

    return float(
        sum(weight * score for weight, score in components)
        / total_weight
    )

def canonical_tempo_from_grid_hypothesis(
    grid_pulse_bpm,
    meter_candidate,
):
    """
    Convierte la velocidad de la cuadrícula a BPM notados.

    Para 6/8 usa como unidad convencional la negra con puntillo.
    Para los compases /4 usa la negra.
    """
    if (
        grid_pulse_bpm is None
        or not np.isfinite(grid_pulse_bpm)
        or grid_pulse_bpm <= 0
    ):
        return None

    numerator = int(meter_candidate["numerator"])
    denominator = int(meter_candidate["denominator"])
    pulses_per_bar = int(meter_candidate["beatsPerBar"])

    if pulses_per_bar <= 0:
        return None

    bar_quarter_length = (
        numerator * 4.0 / denominator
    )

    grid_unit_quarter_length = (
        bar_quarter_length / pulses_per_bar
    )

    declared_unit_length = float(
        meter_candidate["beatUnitQuarterLength"]
    )

    if not np.isclose(
        grid_unit_quarter_length,
        declared_unit_length,
        rtol=0.05,
        atol=0.01,
    ):
        return None

    if (numerator, denominator) == (6, 8):
        tempo_unit = "dotted_quarter"
        tempo_unit_quarter_length = 1.5
    else:
        tempo_unit = "quarter"
        tempo_unit_quarter_length = 1.0

    canonical_beats_per_bar = (
        bar_quarter_length / tempo_unit_quarter_length
    )

    bar_duration_seconds = (
        pulses_per_bar * 60.0 / float(grid_pulse_bpm)
    )

    if bar_duration_seconds <= 0:
        return None

    canonical_tempo_bpm = (
        canonical_beats_per_bar
        * 60.0
        / bar_duration_seconds
    )

    if (
        not np.isfinite(canonical_tempo_bpm)
        or canonical_tempo_bpm <= 0
    ):
        return None

    return {
        "tempoBpm": float(canonical_tempo_bpm),
        "tempoUnit": tempo_unit,
        "gridPulseBpm": float(grid_pulse_bpm),
        "gridPulseUnit": meter_candidate["beatUnit"],
        "gridPulsesPerBar": pulses_per_bar,
        "barDurationSeconds": float(bar_duration_seconds),
    }

def select_joint_tempo_and_meter(
    tempo_candidates,
    source_grids,
    note_starts,
    onset_envelope,
    audio_path,
    sr,
    bar_processor,
):
    """
    Evalúa tempo notado, unidad del pulso y compás.

    La alineación MIDI queda como dato de diagnóstico. No suma puntos
    para elegir el tempo porque las cuadrículas más densas pueden acercar
    artificialmente más notas.
    """
    pair_results = {}

    for target_tempo in tempo_candidates:
        if (
            target_tempo is None
            or not np.isfinite(target_tempo)
            or target_tempo <= 0
        ):
            continue

        for source_name, source_grid in source_grids.items():
            candidate_grids = make_grids_for_candidate_tempo(
                source_grid,
                target_tempo,
            )

            for candidate_grid in candidate_grids:
                grid_pulse_bpm = grid_tempo(candidate_grid)

                if (
                    grid_pulse_bpm is None
                    or grid_pulse_bpm <= 0
                ):
                    continue

                audio_score = autocorrelation_score_for_tempo(
                    onset_envelope,
                    sr,
                    grid_pulse_bpm,
                )

                meter_result = estimate_meter_for_grid(
                    audio_path=audio_path,
                    beat_times=candidate_grid,
                    onset_envelope=onset_envelope,
                    sr=sr,
                    bar_processor=bar_processor,
                )

                pulse_hypotheses = meter_result.get(
                    "pulseHypotheses",
                    [],
                )

                if not pulse_hypotheses:
                    continue

                midi_score = midi_alignment_score(
                    note_starts,
                    candidate_grid,
                )

                for meter_candidate in pulse_hypotheses:
                    tempo_interpretation = (
                        canonical_tempo_from_grid_hypothesis(
                            grid_pulse_bpm,
                            meter_candidate,
                        )
                    )

                    if tempo_interpretation is None:
                        continue

                    canonical_tempo = round(
                        tempo_interpretation["tempoBpm"],
                        1,
                    )
                    tempo_unit = tempo_interpretation["tempoUnit"]

                    total_score = combine_available_scores(
                        audio_score,
                        midi_score,
                        meter_candidate["score"],
                    )

                    if total_score is None:
                        continue

                    numerator = int(
                        meter_candidate["numerator"]
                    )
                    denominator = int(
                        meter_candidate["denominator"]
                    )

                    hypothesis = {
                        "tempoBpm": canonical_tempo,
                        "tempoUnit": tempo_unit,
                        "gridPulseBpm": round(
                            grid_pulse_bpm,
                            1,
                        ),
                        "gridPulseUnit":
                            meter_candidate["beatUnit"],
                        "gridPulsesPerBar":
                            int(meter_candidate["beatsPerBar"]),
                        "barDurationSeconds": round(
                            tempo_interpretation[
                                "barDurationSeconds"
                            ],
                            4,
                        ),
                        "numerator": numerator,
                        "denominator": denominator,
                        "audioScore": audio_score,
                        "midiScore": midi_score,
                        "meterScore": float(
                            meter_candidate["score"]
                        ),
                        "totalScore": float(total_score),
                        "gridSource": str(source_name),
                        "isSelected": False,
                    }

                    # Cuadrículas de fuentes distintas pueden producir
                    # prácticamente el mismo BPM notado y compás.
                    key = None

                    for existing_key in pair_results:
                        same_signature = (
                            existing_key[1] == tempo_unit
                            and existing_key[2] == numerator
                            and existing_key[3] == denominator
                        )

                        tempo_difference = abs(
                            existing_key[0] - canonical_tempo
                        ) / max(existing_key[0], 1e-9)

                        if (
                            same_signature
                            and tempo_difference <= 0.02
                        ):
                            key = existing_key
                            break

                    if key is None:
                        key = (
                            canonical_tempo,
                            tempo_unit,
                            numerator,
                            denominator,
                        )

                    previous = pair_results.get(key)

                    if (
                        previous is None
                        or hypothesis["totalScore"]
                        > previous["hypothesis"]["totalScore"]
                    ):
                        pair_results[key] = {
                            "hypothesis": hypothesis,
                            "meterCandidates":
                                meter_result["candidates"],
                        }

    hypotheses = [
        value["hypothesis"]
        for value in pair_results.values()
    ]

    hypotheses.sort(
        key=lambda item: item["totalScore"],
        reverse=True,
    )

    empty_result = {
        "selectedTempoBpm": None,
        "selectedTempoUnit": None,
        "selectedPulseBpm": None,
        "selectedPulseUnit": None,
        "selectedGridPulsesPerBar": None,
        "tempoConfidence": None,
        "tempoDecisionReason": "no_valid_tempo_hypotheses",
        "selectedMeter": None,
        "meterConfidence": None,
        "meterDecisionReason": "no_valid_meter_hypotheses",
        "tempoHypotheses": [],
        "selectedMeterCandidates": [],
    }

    if not hypotheses:
        return empty_result

    best_score_by_tempo = {}

    for item in hypotheses:
        tempo_key = (
            item["tempoBpm"],
            item["tempoUnit"],
        )

        best_score_by_tempo[tempo_key] = max(
            best_score_by_tempo.get(tempo_key, -1.0),
            item["totalScore"],
        )

    ranked_tempos = sorted(
        best_score_by_tempo.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_tempo = None
    selected_tempo_unit = None

    if len(ranked_tempos) == 1:
        selected_tempo, selected_tempo_unit = ranked_tempos[0][0]
        tempo_margin = 1.0
        tempo_reason = "only_one_valid_tempo_hypothesis"
    else:
        tempo_margin = (
            ranked_tempos[0][1] - ranked_tempos[1][1]
        )

        if tempo_margin >= MIN_TEMPO_HYPOTHESIS_MARGIN:
            selected_tempo, selected_tempo_unit = (
                ranked_tempos[0][0]
            )
            tempo_reason = "joint_audio_meter_score"
        else:
            tempo_reason = "tempo_candidates_are_too_close"

    selected_meter = None
    meter_margin = None
    selected_meter_candidates = []
    selected_pulse_bpm = None
    selected_pulse_unit = None
    selected_grid_pulses_per_bar = None

    if selected_tempo is None:
        meter_reason = "tempo_is_ambiguous"
    else:
        matching_items = [
            item
            for item in hypotheses
            if (
                item["tempoBpm"] == selected_tempo
                and item["tempoUnit"] == selected_tempo_unit
            )
        ]

        scores_by_meter = {}

        for item in matching_items:
            meter_key = (
                item["numerator"],
                item["denominator"],
            )

            current = scores_by_meter.get(meter_key)

            if (
                current is None
                or item["totalScore"] > current["score"]
            ):
                result_for_pair = next(
                    (
                        value
                        for value in pair_results.values()
                        if value["hypothesis"] is item
                    ),
                    None,
                )

                scores_by_meter[meter_key] = {
                    "score": item["totalScore"],
                    "hypothesis": item,
                    "meterCandidates": (
                        result_for_pair["meterCandidates"]
                        if result_for_pair
                        else []
                    ),
                }

        ranked_meters = sorted(
            scores_by_meter.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        if len(ranked_meters) == 1:
            (numerator, denominator), best = ranked_meters[0]
            selected_meter = {
                "numerator": numerator,
                "denominator": denominator,
            }
            meter_margin = 1.0
            selected_meter_candidates = best["meterCandidates"]
            meter_reason = "one_meter_hypothesis_for_selected_tempo"

        elif ranked_meters:
            meter_margin = (
                ranked_meters[0][1]["score"]
                - ranked_meters[1][1]["score"]
            )

            if meter_margin >= MIN_JOINT_METER_MARGIN:
                (numerator, denominator), best = ranked_meters[0]
                selected_meter = {
                    "numerator": numerator,
                    "denominator": denominator,
                }
                selected_meter_candidates = best["meterCandidates"]
                meter_reason = "joint_tempo_meter_score"
            else:
                meter_reason = "meter_candidates_are_too_close"
        else:
            meter_reason = "no_meter_for_selected_tempo"

    matching_tempo_items = [
        item
        for item in hypotheses
        if (
            selected_tempo is not None
            and item["tempoBpm"] == selected_tempo
            and item["tempoUnit"] == selected_tempo_unit
            and (
                selected_meter is None
                or (
                    item["numerator"] == selected_meter["numerator"]
                    and item["denominator"] == selected_meter["denominator"]
                )
            )
        )
    ]

    if matching_tempo_items:
        best_tempo_item = max(
            matching_tempo_items,
            key=lambda item: item["totalScore"],
        )
        selected_pulse_bpm = best_tempo_item["gridPulseBpm"]
        selected_pulse_unit = best_tempo_item["gridPulseUnit"]
        selected_grid_pulses_per_bar = (
            best_tempo_item["gridPulsesPerBar"]
        )

    for item in hypotheses:
        item["isSelected"] = bool(
            selected_tempo is not None
            and selected_meter is not None
            and item["tempoBpm"] == selected_tempo
            and item["tempoUnit"] == selected_tempo_unit
            and item["numerator"]
                == selected_meter["numerator"]
            and item["denominator"]
                == selected_meter["denominator"]
        )

    return {
        "selectedTempoBpm": selected_tempo,
        "selectedTempoUnit": selected_tempo_unit,
        "selectedPulseBpm": selected_pulse_bpm,
        "selectedPulseUnit": selected_pulse_unit,
        "selectedGridPulsesPerBar":
            selected_grid_pulses_per_bar,
        "tempoConfidence": tempo_margin,
        "tempoDecisionReason": tempo_reason,
        "selectedMeter": selected_meter,
        "meterConfidence": meter_margin,
        "meterDecisionReason": meter_reason,
        "tempoHypotheses": hypotheses,
        "selectedMeterCandidates": selected_meter_candidates,
    }

TEMPO_ESTIMATOR_AGREEMENT_TOLERANCE = 0.05


def estimate_numeric_tempo(librosa_tempo, madmom_tempo):
    """
    Estima el valor numérico del tempo sin decidir la unidad de nota
    ni el compás.

    Si ambos rastreadores coinciden dentro del 5 %, devuelve su promedio.
    Si solo uno está disponible, devuelve ese valor y lo identifica como
    una estimación de una sola fuente.
    Si ambos discrepan, no fabrica un consenso.
    """
    def valid_tempo(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(numeric) or numeric <= 0:
            return None

        return numeric

    librosa_value = valid_tempo(librosa_tempo)
    madmom_value = valid_tempo(madmom_tempo)

    if librosa_value is not None and madmom_value is not None:
        relative_difference = (
            abs(librosa_value - madmom_value)
            / max(librosa_value, madmom_value)
        )

        agreement = float(
            np.clip(1.0 - relative_difference, 0.0, 1.0)
        )

        if relative_difference <= TEMPO_ESTIMATOR_AGREEMENT_TOLERANCE:
            return {
                "estimatedTempoBpm": round(
                    (librosa_value + madmom_value) / 2.0,
                    1,
                ),
                "tempoEstimateSource": "librosa_madmom_consensus",
                "tempoEstimatorAgreement": round(agreement, 4),
            }

        return {
            "estimatedTempoBpm": None,
            "tempoEstimateSource": "estimators_disagree",
            "tempoEstimatorAgreement": round(agreement, 4),
        }

    if librosa_value is not None:
        return {
            "estimatedTempoBpm": round(librosa_value, 1),
            "tempoEstimateSource": "librosa_only",
            "tempoEstimatorAgreement": None,
        }

    if madmom_value is not None:
        return {
            "estimatedTempoBpm": round(madmom_value, 1),
            "tempoEstimateSource": "madmom_only",
            "tempoEstimatorAgreement": None,
        }

    return {
        "estimatedTempoBpm": None,
        "tempoEstimateSource": "no_estimate",
        "tempoEstimatorAgreement": None,
    }