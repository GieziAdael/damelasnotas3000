using System;
using System.Collections.Generic;
using System.Text;

namespace Damelasnotas3000.Models
{
    public sealed record MidiCleanupResult(
        int TotalNotes,
        int SupportedCount,
        int ReviewCount,
        int InconclusiveCount,
        IReadOnlyList<NoteAssessment> Notes,
        int RemovedCount = 0,
        string? CleanedMidiPath = null,
        string? ReportPath = null);
}
