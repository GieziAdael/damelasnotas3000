using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public record MusicAnalysisResult(
        double? TempoBpm,
        int? BeatCount,
        int? TimeSignatureNumerator,
        int? TimeSignatureDenominator,
        string? Key,
        double? Confidence,
        double? MadmomTempoBpm = null,
        IReadOnlyList<double>? TempoCandidatesBpm = null,
        IReadOnlyList<MeterCandidate>? MeterCandidates = null,
        double? MeterConfidence = null,
        IReadOnlyList<double>? LibrosaBeatTimes = null,
        IReadOnlyList<double>? MadmomBeatTimes = null,
        BeatGridComparison? BeatGridComparison = null,
        string? MeterDecisionReason = null,
        double? SelectedTempoBpm = null,
        double? TempoConfidence = null,
        string? TempoDecisionReason = null,
        IReadOnlyList<TempoHypothesis>? TempoHypotheses = null,
        string? SelectedTempoUnit = null,
        double? SelectedPulseBpm = null,
        string? SelectedPulseUnit = null,
        int? SelectedGridPulsesPerBar = null,
        double? EstimatedTempoBpm = null,
        string? TempoEstimateSource = null,
        double? TempoEstimatorAgreement = null);
}
