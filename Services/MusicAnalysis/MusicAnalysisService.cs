using System.Diagnostics;
using System.Text.Json;
using Damelasnotas3000.Models;
using Damelasnotas3000.Services;

namespace Damelasnotas3000.Services.MusicAnalysis;

public class MusicAnalysisService
{
    public async Task<MusicAnalysisResult> AnalyzeAsync(
        string audioFilePath,
        string midiPath,
        string outputDirectory)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(audioFilePath) ||
            !File.Exists(audioFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el audio para analizar.",
                audioFilePath);
        }

        if (string.IsNullOrWhiteSpace(midiPath) ||
            !File.Exists(midiPath))
        {
            throw new FileNotFoundException(
                "No se encontró el MIDI para analizar.",
                midiPath);
        }

        Directory.CreateDirectory(outputDirectory);

        string analysisWavPath = Path.Combine(
            outputDirectory,
            "audio-para-analisis.wav");

        File.Copy(
            audioFilePath,
            analysisWavPath,
            overwrite: true);

        string pythonExe = Path.Combine(
            FileSystem.AppDataDirectory,
            "Magenta",
            "Runtime",
            "Scripts",
            "python.exe");

        if (!File.Exists(pythonExe))
        {
            throw new FileNotFoundException(
                "No se encontró el Python de Magenta.",
                pythonExe);
        }

        string[] scriptNames =
        {
            "meter_analysis.py",
            "audio_features.py",
            "beat_grid.py",
            "beat_tracking.py",
            "meter_scoring.py",
            "midi_diagnostics.py",
            "tempo_meter_selection.py"
        };

        foreach (string scriptName in scriptNames)
        {
            await using Stream source =
                await FileSystem.OpenAppPackageFileAsync(
                    $"MusicAnalysis/{scriptName}");
            await using FileStream destination = File.Create(
                Path.Combine(outputDirectory, scriptName));

            await source.CopyToAsync(destination);
        }

        string scriptPath = Path.Combine(
            outputDirectory,
            "meter_analysis.py");

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.Environment["AUDIO_INPUT"] = analysisWavPath;
        startInfo.Environment["MIDI_INPUT"] = midiPath;

        // Se ejecuta el archivo Python directamente, no con "-c".
        startInfo.ArgumentList.Add(scriptPath);

        using var process = new Process
        {
            StartInfo = startInfo
        };

        process.Start();

        Task<string> stdoutTask =
            process.StandardOutput.ReadToEndAsync();

        Task<string> stderrTask =
            process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        await AnalysisDebugLog.AppendProcessOutputAsync(
            outputDirectory,
            "ANÁLISIS DE AUDIO: LIBROSA, MADMOM, COMPÁS Y PULSOS",
            process.ExitCode,
            stdout,
            stderr);

        Debug.WriteLine(stdout);
        Debug.WriteLine(stderr);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "Falló el análisis del audio." +
                $"{Environment.NewLine}{stderr}" +
                $"{Environment.NewLine}{stdout}");
        }

        string? jsonLine = stdout
            .Split(
                new[] { "\r\n", "\n" },
                StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault(line =>
                line.StartsWith(
                    "ANALYSIS_JSON:",
                    StringComparison.Ordinal));

        if (jsonLine is null)
        {
            throw new InvalidOperationException(
                "El analizador no devolvió un resultado válido." +
                $"{Environment.NewLine}{stdout}");
        }

        TempoAnalysisOutput? result =
            JsonSerializer.Deserialize<TempoAnalysisOutput>(
                jsonLine["ANALYSIS_JSON:".Length..],
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

        if (result is null)
        {
            throw new InvalidOperationException(
                "No se pudo leer el resultado del análisis.");
        }

        await AnalysisDebugLog.AppendSectionAsync(
            outputDirectory,
            "ANÁLISIS INTERPRETADO POR LA APLICACIÓN (JSON)",
            JsonSerializer.Serialize(
                result,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));

        Debug.WriteLine(
            $"Pulsos Librosa: {result.LibrosaBeatTimes?.Count ?? 0}");

        Debug.WriteLine(
            $"Pulsos Madmom: {result.MadmomBeatTimes?.Count ?? 0}");

        Debug.WriteLine(
            $"Relación de cuadrículas: " +
            $"{result.BeatGridComparison?.RateRelation}");

        Debug.WriteLine(
            $"Motivo del compás: {result.MeterDecisionReason}");

        return new MusicAnalysisResult(
            TempoBpm: result.TempoBpm,
            BeatCount: result.BeatCount,
            TimeSignatureNumerator:
                result.TimeSignatureNumerator,
            TimeSignatureDenominator:
                result.TimeSignatureDenominator,
            Key: null,
            Confidence: null,
            MadmomTempoBpm: result.MadmomTempoBpm,
            TempoCandidatesBpm: result.TempoCandidatesBpm,
            MeterCandidates: result.MeterCandidates,
            MeterConfidence: result.MeterConfidence,
            LibrosaBeatTimes: result.LibrosaBeatTimes,
            MadmomBeatTimes: result.MadmomBeatTimes,
            BeatGridComparison: result.BeatGridComparison,
            MeterDecisionReason: result.MeterDecisionReason,
            SelectedTempoBpm: result.SelectedTempoBpm,
            TempoConfidence: result.TempoConfidence,
            TempoDecisionReason: result.TempoDecisionReason,
            TempoHypotheses: result.TempoHypotheses,
            SelectedTempoUnit: result.SelectedTempoUnit,
            SelectedPulseBpm: result.SelectedPulseBpm,
            SelectedPulseUnit: result.SelectedPulseUnit,
            SelectedGridPulsesPerBar:
                result.SelectedGridPulsesPerBar,
            EstimatedTempoBpm: result.EstimatedTempoBpm,
            TempoEstimateSource: result.TempoEstimateSource,
            TempoEstimatorAgreement:
                result.TempoEstimatorAgreement);
#else
        throw new PlatformNotSupportedException(
            "El análisis local está configurado para Windows.");
#endif
    }

    private sealed record TempoAnalysisOutput(
        double? TempoBpm,
        double? MadmomTempoBpm,
        List<double>? TempoCandidatesBpm,
        int? BeatCount,
        int? TimeSignatureNumerator,
        int? TimeSignatureDenominator,
        List<MeterCandidate>? MeterCandidates,
        double? MeterConfidence,
        List<double>? LibrosaBeatTimes,
        List<double>? MadmomBeatTimes,
        BeatGridComparison? BeatGridComparison,
        string? MeterDecisionReason,
        double? SelectedTempoBpm,
        double? TempoConfidence,
        string? TempoDecisionReason,
        List<TempoHypothesis>? TempoHypotheses,
        string? SelectedTempoUnit,
        double? SelectedPulseBpm,
        string? SelectedPulseUnit,
        int? SelectedGridPulsesPerBar,
        double? EstimatedTempoBpm,
        string? TempoEstimateSource,
        double? TempoEstimatorAgreement);
}
