using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record MeterCandidate(
        int Numerator,
        int Denominator,
        double Score,
        int? BeatsPerBar = null,
        string? BeatUnit = null,
        double? BeatUnitQuarterLength = null);
}
