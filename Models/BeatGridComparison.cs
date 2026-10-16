using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record BeatGridComparison(
        double? LibrosaGridTempoBpm,
        double? MadmomGridTempoBpm,
        double? MadmomToLibrosaTempoRatio,
        double? SlowerToFasterPulsePeriodRatio,
        int? ExpectedFastPulsesPerSlowPulse,
        double? MatchedFraction,
        double? MedianAlignmentErrorMs,
        string? RateRelation);
}
