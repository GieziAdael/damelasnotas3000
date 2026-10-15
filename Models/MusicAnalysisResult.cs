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
        double? MeterConfidence = null);
}
