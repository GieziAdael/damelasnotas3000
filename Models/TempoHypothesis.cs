using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record TempoHypothesis(
        double TempoBpm,
        int Numerator,
        int Denominator,
        double? AudioScore,
        double? MidiScore,
        double MeterScore,
        double TotalScore,
        string? GridSource,
        bool IsSelected,
        string? TempoUnit = null,
        double? GridPulseBpm = null,
        string? GridPulseUnit = null,
        int? GridPulsesPerBar = null,
        double? BarDurationSeconds = null);
}
