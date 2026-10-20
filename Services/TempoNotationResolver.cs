using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services;

/// <summary>
/// A tempo mark needs both a numeric rate and a note value. The independent
/// pulse estimate must not be assigned a note value unless the user or the
/// analysis has resolved that unit.
/// </summary>
internal sealed record TempoNotation(
    double BeatsPerMinute,
    string NoteUnit,
    bool UsesEstimatedPulse,
    bool UsesManualBpm);

internal static class TempoNotationResolver
{
    internal static TempoNotation? Resolve(
        MusicAnalysisResult? analysis,
        double? tempoBpmOverride,
        string? tempoUnitOverride)
    {
        if (tempoBpmOverride is double manualBpm)
        {
            if (!IsValidTempo(manualBpm))
                return null;

            // Preserve the existing convention for a manually entered BPM
            // without a note value: interpret it as quarter notes per minute.
            string manualUnit = string.IsNullOrWhiteSpace(tempoUnitOverride)
                ? "quarter"
                : tempoUnitOverride!;

            return new TempoNotation(
                manualBpm,
                manualUnit,
                UsesEstimatedPulse: false,
                UsesManualBpm: true);
        }

        if (!string.IsNullOrWhiteSpace(tempoUnitOverride))
        {
            if (analysis?.EstimatedTempoBpm is double estimatedBpm &&
                IsValidTempo(estimatedBpm))
            {
                // The selected note value supplies the interpretation; the
                // independent estimate supplies only the pulse rate.
                return new TempoNotation(
                    estimatedBpm,
                    tempoUnitOverride!,
                    UsesEstimatedPulse: true,
                    UsesManualBpm: false);
            }

            if (analysis?.SelectedTempoBpm is double selectedBpm &&
                IsValidTempo(selectedBpm) &&
                string.Equals(
                    analysis.SelectedTempoUnit,
                    tempoUnitOverride,
                    StringComparison.Ordinal))
            {
                return new TempoNotation(
                    selectedBpm,
                    tempoUnitOverride!,
                    UsesEstimatedPulse: false,
                    UsesManualBpm: false);
            }

            return null;
        }

        if (analysis?.SelectedTempoBpm is double analyzedBpm &&
            IsValidTempo(analyzedBpm) &&
            !string.IsNullOrWhiteSpace(analysis.SelectedTempoUnit))
        {
            return new TempoNotation(
                analyzedBpm,
                analysis.SelectedTempoUnit!,
                UsesEstimatedPulse: false,
                UsesManualBpm: false);
        }

        // Without a resolved note value, retain the numeric pulse estimate in
        // diagnostics but do not invent a tempo mark for the MusicXML.
        return null;
    }

    private static bool IsValidTempo(double bpm) =>
        double.IsFinite(bpm) && bpm > 0;
}
