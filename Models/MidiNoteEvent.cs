using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record MidiNoteEvent(
        int EventIndex,
        int InstrumentIndex,
        string? InstrumentName,
        int Program,
        bool IsDrum,
        int Pitch,
        string PitchName,
        double StartSeconds,
        double DurationSeconds,
        int Velocity);
}
