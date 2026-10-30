using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services;

public class MidiToMusicXmlService
{
    public async Task<MusicXmlConversionResult> ConvertAsync(
        string midiFilePath,
        string outputDirectory,
        MusicAnalysisResult analysis,
        int? meterNumeratorOverride = null,
        int? meterDenominatorOverride = null,
        double? tempoBpmOverride = null,
        string? tempoUnitOverride = null)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(midiFilePath) ||
            !File.Exists(midiFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el archivo MIDI.",
                midiFilePath);
        }

        if (analysis is null)
            throw new ArgumentNullException(nameof(analysis));

        if (meterNumeratorOverride.HasValue !=
            meterDenominatorOverride.HasValue)
        {
            throw new ArgumentException(
                "Debes indicar numerador y denominador del compás.");
        }

        if (meterNumeratorOverride is int numeratorOverride &&
            (numeratorOverride < 1 || numeratorOverride > 32))
        {
            throw new ArgumentOutOfRangeException(
                nameof(meterNumeratorOverride),
                "El numerador del compás debe estar entre 1 y 32.");
        }

        if (meterDenominatorOverride is int denominatorOverride &&
            (denominatorOverride < 1 || denominatorOverride > 32 ||
             (denominatorOverride & (denominatorOverride - 1)) != 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(meterDenominatorOverride),
                "El denominador del compás debe ser una potencia de 2.");
        }

        if (tempoBpmOverride is double manualTempo &&
            (!double.IsFinite(manualTempo) || manualTempo <= 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(tempoBpmOverride),
                "El tempo debe ser un número mayor que cero.");
        }

        Directory.CreateDirectory(outputDirectory);

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

        string converterPackagesDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreConverterPackages");

        string musicXmlPath = Path.Combine(
            outputDirectory,
            "grabacion.musicxml");
        string timingMapPath = Path.Combine(
            outputDirectory,
            "score_timing_map.json");

        string scriptPath = Path.Combine(
            outputDirectory,
            "midi_to_musicxml.py");

        await using (Stream source =
            await FileSystem.OpenAppPackageFileAsync(
                "MusicAnalysis/midi_to_musicxml.py"))
        await using (FileStream destination =
            File.Create(scriptPath))
        {
            await source.CopyToAsync(destination);
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.Environment["PYTHONPATH"] =
            converterPackagesDirectory;
        startInfo.Environment["MIDI_INPUT"] = midiFilePath;
        startInfo.Environment["XML_OUTPUT"] = musicXmlPath;
        startInfo.Environment["TIMING_MAP_OUTPUT"] = timingMapPath;
        startInfo.Environment.Remove("METER_PHASE_INDEX");
        startInfo.Environment.Remove("METER_PHASE_DOWNBEAT_SECONDS");
        startInfo.Environment.Remove("METER_PHASE_SOURCE");

        int? meterNumerator =
            meterNumeratorOverride ?? analysis.TimeSignatureNumerator;
        int? meterDenominator =
            meterDenominatorOverride ?? analysis.TimeSignatureDenominator;

        if (meterNumerator is int numerator &&
            meterDenominator is int denominator)
        {
            startInfo.Environment["METER_NUMERATOR"] =
                numerator.ToString(CultureInfo.InvariantCulture);
            startInfo.Environment["METER_DENOMINATOR"] =
                denominator.ToString(CultureInfo.InvariantCulture);

            // Reutiliza la fase métrica detectada para que el conversor
            // sitúe los compases según los acentos del audio, no según el
            // primer ataque que Magenta haya producido.
            var phaseCandidate = analysis.MeterCandidates?
                .Where(candidate =>
                    candidate.Numerator == numerator &&
                    candidate.Denominator == denominator &&
                    candidate.PhaseIndex.HasValue)
                .OrderByDescending(candidate => candidate.Score)
                .FirstOrDefault();
            var selectedHypothesis = analysis.TempoHypotheses?
                .Where(hypothesis =>
                    hypothesis.Numerator == numerator &&
                    hypothesis.Denominator == denominator)
                .OrderByDescending(hypothesis => hypothesis.IsSelected)
                .ThenByDescending(hypothesis => hypothesis.TotalScore)
                .FirstOrDefault();

            IReadOnlyList<double>? phaseBeatTimes =
                string.Equals(
                    selectedHypothesis?.GridSource,
                    "madmom",
                    StringComparison.OrdinalIgnoreCase)
                    ? analysis.MadmomBeatTimes
                    : analysis.LibrosaBeatTimes;

            if (phaseCandidate?.PhaseIndex is int phaseIndex &&
                phaseBeatTimes is { Count: > 0 } &&
                phaseIndex >= 0 && phaseIndex < phaseBeatTimes.Count)
            {
                startInfo.Environment["METER_PHASE_INDEX"] =
                    phaseIndex.ToString(CultureInfo.InvariantCulture);
                startInfo.Environment["METER_PHASE_DOWNBEAT_SECONDS"] =
                    phaseBeatTimes[phaseIndex].ToString(
                        "0.########",
                        CultureInfo.InvariantCulture);
                startInfo.Environment["METER_PHASE_SOURCE"] =
                    selectedHypothesis?.GridSource ?? "librosa";
            }
        }

        TempoNotation? tempoNotation = TempoNotationResolver.Resolve(
            analysis,
            tempoBpmOverride,
            tempoUnitOverride);

        if (tempoNotation is not null)
        {
            startInfo.Environment["TEMPO_BPM"] =
                tempoNotation.BeatsPerMinute.ToString(
                    "0.###",
                    CultureInfo.InvariantCulture);
            startInfo.Environment["TEMPO_UNIT"] =
                tempoNotation.NoteUnit;
        }

        startInfo.ArgumentList.Add(scriptPath);

        using var process = new Process { StartInfo = startInfo };
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
            "CONVERSIÓN DEL MIDI A MUSICXML",
            process.ExitCode,
            stdout,
            stderr);

        Debug.WriteLine(stdout);
        Debug.WriteLine(stderr);

        if (process.ExitCode != 0 || !File.Exists(musicXmlPath))
        {
            throw new InvalidOperationException(
                "No se pudo convertir el MIDI a MusicXML." +
                $"{Environment.NewLine}{stderr}" +
                $"{Environment.NewLine}{stdout}");
        }

        if (!File.Exists(timingMapPath))
        {
            await AnalysisDebugLog.AppendFileAsync(
                outputDirectory,
                "MAPA DE SINCRONIZACIÓN DE LA PARTITURA (JSON)",
                timingMapPath);
            throw new InvalidOperationException(
                "El conversor no generó el mapa de sincronización de la partitura.");
        }

        await AnalysisDebugLog.AppendFileAsync(
            outputDirectory,
            "MAPA DE SINCRONIZACIÓN DE LA PARTITURA (JSON)",
            timingMapPath);

        await using FileStream timingMapStream = File.OpenRead(timingMapPath);
        ScoreTimingMap? timingMap = await JsonSerializer.DeserializeAsync<ScoreTimingMap>(
            timingMapStream,
            new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

        if (timingMap is null ||
            !double.IsFinite(timingMap.QuarterLengthScale) ||
            timingMap.QuarterLengthScale <= 0 ||
            !double.IsFinite(timingMap.TrimmedQuarterLengths) ||
            timingMap.TrimmedQuarterLengths < 0 ||
            (timingMap.OnsetOriginCorrectionQL is double onsetCorrection &&
             (!double.IsFinite(onsetCorrection) || onsetCorrection < 0)) ||
            timingMap.QuantizationSubdivisions is not (1 or 2 or 3 or 4 or 6 or 8))
        {
            throw new InvalidOperationException(
                "El mapa de sincronización de la partitura no es válido.");
        }

        return new MusicXmlConversionResult(musicXmlPath, timingMap);
#else
        throw new PlatformNotSupportedException(
            "La conversión local está configurada para Windows.");
#endif
    }
}

public sealed record ScoreTimingMap(
    double QuarterLengthScale,
    double TrimmedQuarterLengths,
    int QuantizationSubdivisions,
    double? OnsetOriginCorrectionQL = null);

public sealed record MusicXmlConversionResult(
    string MusicXmlPath,
    ScoreTimingMap TimingMap);
