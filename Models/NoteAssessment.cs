using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record NoteAssessment(
        MidiNoteEvent Note,
        double? PitchContrastDb,
        double? OnsetRiseDb,
        string Status,
        string Reason);
}
