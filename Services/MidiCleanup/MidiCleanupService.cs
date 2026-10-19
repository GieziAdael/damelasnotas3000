using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services.MidiCleanup;

public sealed class MidiCleanupService
{
    public async Task<MidiCleanupResult> CleanAsync(
        string audioFilePath,
        string midiFilePath,
        string outputDirectory,
        MusicAnalysisResult? analysis = null,
        int? meterNumeratorOverride = null,
        int? meterDenominatorOverride = null,
        double? tempoBpmOverride = null,
        string? tempoUnitOverride = null)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(audioFilePath) ||
            !File.Exists(audioFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el audio para evaluar las notas.",
                audioFilePath);
        }

        if (string.IsNullOrWhiteSpace(midiFilePath) ||
            !File.Exists(midiFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el MIDI para limpiar.",
                midiFilePath);
        }

        Directory.CreateDirectory(outputDirectory);

        string magentaPythonExe = Path.Combine(
            FileSystem.AppDataDirectory,
            "Magenta",
            "Runtime",
            "Scripts",
            "python.exe");

        if (!File.Exists(magentaPythonExe))
        {
            throw new FileNotFoundException(
                "No se encontró el Python de Magenta.",
                magentaPythonExe);
        }

        string cleanupScriptPath = Path.Combine(
            outputDirectory,
            "clean_midi.py");

        await CopyPackageFileAsync(
            "MidiCleanup/clean_midi.py",
            cleanupScriptPath);

        string cleanedMidiPath = Path.Combine(
            outputDirectory,
            "grabacion_limpia.mid");

        string diagnosticReportPath = Path.Combine(
            outputDirectory,
            "midi_cleanup_diagnostic.json");

        var cleanupStartInfo = CreatePythonStartInfo(
            magentaPythonExe,
            cleanupScriptPath,
            outputDirectory);

        cleanupStartInfo.Environment["AUDIO_INPUT"] = audioFilePath;
        cleanupStartInfo.Environment["MIDI_INPUT"] = midiFilePath;
        cleanupStartInfo.Environment["CLEANED_MIDI_OUTPUT"] = cleanedMidiPath;

        ProcessOutput cleanupOutput = await RunProcessAsync(cleanupStartInfo);

        Debug.WriteLine("Limpieza MIDI:");
        Debug.WriteLine(cleanupOutput.StandardOutput);
        WriteStandardError(cleanupOutput.StandardError);

        if (cleanupOutput.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "Falló la limpieza de eventos MIDI." +
                $"{Environment.NewLine}{cleanupOutput.StandardError}" +
                $"{Environment.NewLine}{cleanupOutput.StandardOutput}");
        }

        if (!File.Exists(cleanedMidiPath) ||
            new FileInfo(cleanedMidiPath).Length == 0)
        {
            throw new InvalidOperationException(
                "El limpiador no generó un MIDI válido." +
                $"{Environment.NewLine}{cleanupOutput.StandardOutput}");
        }

        string? jsonLine = cleanupOutput.StandardOutput
            .Split(
                new[] { "\r\n", "\n" },
                StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault(line =>
                line.StartsWith(
                    "CLEANUP_JSON:",
                    StringComparison.Ordinal));

        if (jsonLine is null)
        {
            throw new InvalidOperationException(
                "El limpiador no devolvió el JSON esperado." +
                $"{Environment.NewLine}{cleanupOutput.StandardOutput}");
        }

        string cleanupJson = jsonLine["CLEANUP_JSON:".Length..];
        MidiCleanupResult? cleanupResult =
            JsonSerializer.Deserialize<MidiCleanupResult>(
                cleanupJson,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

        if (cleanupResult is null)
        {
            throw new InvalidOperationException(
                "No se pudo leer el resultado de la limpieza MIDI.");
        }

        MidiCleanupResult resultWithPaths = cleanupResult with
        {
            CleanedMidiPath = cleanedMidiPath,
            ReportPath = diagnosticReportPath
        };

        JsonObject diagnostic =
            JsonSerializer.SerializeToNode(
                resultWithPaths,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                })?.AsObject()
            ?? new JsonObject();

        JsonNode comparison = await RunComparisonIfAvailableAsync(
            audioFilePath,
            midiFilePath,
            outputDirectory,
            analysis,
            meterNumeratorOverride,
            meterDenominatorOverride,
            tempoBpmOverride,
            tempoUnitOverride);

        diagnostic["transcriptionComparison"] = comparison;

        await File.WriteAllTextAsync(
            diagnosticReportPath,
            diagnostic.ToJsonString(new JsonSerializerOptions
            {
                WriteIndented = true
            }));

        return resultWithPaths;
#else
        throw new PlatformNotSupportedException(
            "La limpieza local de eventos está configurada para Windows.");
#endif
    }

#if WINDOWS
    private static async Task<JsonNode> RunComparisonIfAvailableAsync(
        string audioFilePath,
        string magentaMidiPath,
        string outputDirectory,
        MusicAnalysisResult? analysis,
        int? meterNumeratorOverride,
        int? meterDenominatorOverride,
        double? tempoBpmOverride,
        string? tempoUnitOverride)
    {
        string? basicPitchPythonExe = FindBasicPitchPythonExecutable();
        if (basicPitchPythonExe is null)
        {
            Debug.WriteLine(
                "Comparación de transcriptores omitida: no se encontró " +
                "Python con Basic Pitch.");

            return new JsonObject
            {
                ["status"] = "unavailable",
                ["reason"] =
                    "No se encontró .venv\\Scripts\\python.exe con Basic Pitch. " +
                    "La limpieza Magenta terminó normalmente."
            };
        }

        string comparisonScriptPath = Path.Combine(
            outputDirectory,
            "compare_transcriptions.py");

        try
        {
            await CopyPackageFileAsync(
                "MusicAnalysis/compare_transcriptions.py",
                comparisonScriptPath);

            string comparisonReportPath = Path.Combine(
                outputDirectory,
                "transcription_comparison.json");

            var startInfo = CreatePythonStartInfo(
                basicPitchPythonExe,
                comparisonScriptPath,
                outputDirectory);

            startInfo.Environment["AUDIO_INPUT"] = audioFilePath;
            startInfo.Environment["MAGENTA_MIDI_INPUT"] = magentaMidiPath;
            startInfo.Environment["COMPARISON_OUTPUT_DIRECTORY"] = outputDirectory;
            startInfo.Environment["BASIC_PITCH_MIDI_OUTPUT"] = Path.Combine(
                outputDirectory,
                "grabacion_basic_pitch.mid");
            startInfo.Environment["CONSENSUS_MIDI_OUTPUT"] = Path.Combine(
                outputDirectory,
                "grabacion_consenso.mid");
            startInfo.Environment["COMPARISON_REPORT_OUTPUT"] = comparisonReportPath;
            startInfo.Environment["COMPARISON_CSV_OUTPUT"] = Path.Combine(
                outputDirectory,
                "transcription_comparison.csv");
            startInfo.Environment["OCTAVE_REVIEW_CSV_OUTPUT"] = Path.Combine(
                outputDirectory,
                "octave_discrepancy_review.csv");

            int? scoreNumerator =
                meterNumeratorOverride ?? analysis?.TimeSignatureNumerator;
            int? scoreDenominator =
                meterDenominatorOverride ?? analysis?.TimeSignatureDenominator;
            if (scoreNumerator is int numerator &&
                scoreDenominator is int denominator)
            {
                startInfo.Environment["SCORE_METER_NUMERATOR"] =
                    numerator.ToString(CultureInfo.InvariantCulture);
                startInfo.Environment["SCORE_METER_DENOMINATOR"] =
                    denominator.ToString(CultureInfo.InvariantCulture);
            }

            double? scoreTempo = tempoBpmOverride ??
                analysis?.SelectedTempoBpm ??
                analysis?.MadmomTempoBpm ??
                analysis?.TempoBpm;
            if (scoreTempo is double bpm &&
                double.IsFinite(bpm) && bpm > 0)
            {
                startInfo.Environment["SCORE_TEMPO_BPM"] =
                    bpm.ToString("0.###", CultureInfo.InvariantCulture);
                startInfo.Environment["SCORE_TEMPO_UNIT"] =
                    tempoUnitOverride ??
                    (tempoBpmOverride.HasValue
                        ? "quarter"
                        : analysis?.SelectedTempoUnit ?? "quarter");
            }

            ProcessOutput comparisonOutput = await RunProcessAsync(
                startInfo,
                TimeSpan.FromMinutes(5));

            Debug.WriteLine("Comparación Magenta / Basic Pitch:");
            Debug.WriteLine(comparisonOutput.StandardOutput);
            WriteStandardError(comparisonOutput.StandardError);

            if (comparisonOutput.ExitCode == 0 &&
                File.Exists(comparisonReportPath))
            {
                JsonNode? report = JsonNode.Parse(
                    await File.ReadAllTextAsync(comparisonReportPath));

                if (report is not null)
                {
                    return report;
                }
            }

            return new JsonObject
            {
                ["status"] = "failed",
                ["reportPath"] = comparisonReportPath,
                ["exitCode"] = comparisonOutput.ExitCode,
                ["error"] = TrimForDiagnostic(
                    comparisonOutput.StandardError),
                ["output"] = TrimForDiagnostic(
                    comparisonOutput.StandardOutput)
            };
        }
        catch (Exception exception)
        {
            // Basic Pitch es una segunda opinion. Si falta una dependencia o
            // el modelo falla, se conserva el resultado del flujo Magenta.
            Debug.WriteLine(
                "Falló la comparación opcional con Basic Pitch: " + exception);

            return new JsonObject
            {
                ["status"] = "failed",
                ["error"] = TrimForDiagnostic(exception.ToString())
            };
        }
    }

    private static string? FindBasicPitchPythonExecutable()
    {
        string? configuredPath =
            Environment.GetEnvironmentVariable("BASIC_PITCH_PYTHON");

        if (!string.IsNullOrWhiteSpace(configuredPath) &&
            File.Exists(configuredPath))
        {
            return configuredPath;
        }

        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(
                directory.FullName,
                ".venv",
                "Scripts",
                "python.exe");

            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        string appDataCandidate = Path.Combine(
            FileSystem.AppDataDirectory,
            "BasicPitch",
            "Scripts",
            "python.exe");

        return File.Exists(appDataCandidate)
            ? appDataCandidate
            : null;
    }

    private static ProcessStartInfo CreatePythonStartInfo(
        string pythonExe,
        string scriptPath,
        string workingDirectory)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add("-u");
        startInfo.ArgumentList.Add(scriptPath);
        return startInfo;
    }

    private static async Task<ProcessOutput> RunProcessAsync(
        ProcessStartInfo startInfo,
        TimeSpan? timeout = null)
    {
        using var process = new Process
        {
            StartInfo = startInfo
        };

        if (!process.Start())
        {
            throw new InvalidOperationException(
                $"No se pudo iniciar el proceso {startInfo.FileName}.");
        }

        Task<string> stdoutTask =
            process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask =
            process.StandardError.ReadToEndAsync();

        Task waitTask = process.WaitForExitAsync();
        bool timedOut = false;

        if (timeout is TimeSpan maximumDuration)
        {
            Task completedTask = await Task.WhenAny(
                waitTask,
                Task.Delay(maximumDuration));

            if (completedTask != waitTask)
            {
                try
                {
                    if (!process.HasExited)
                    {
                        process.Kill(entireProcessTree: true);
                        timedOut = true;
                    }
                }
                catch (InvalidOperationException)
                {
                    // El proceso terminó justo al vencer el límite.
                }
            }
        }

        await waitTask;

        string standardOutput = await stdoutTask;
        string standardError = await stderrTask;

        if (timedOut)
        {
            standardError = string.Join(
                Environment.NewLine,
                standardError,
                $"La comparación superó el límite de {timeout!.Value.TotalMinutes:F0} minutos.");
            return new ProcessOutput(-1, standardOutput, standardError);
        }

        return new ProcessOutput(
            process.ExitCode,
            standardOutput,
            standardError);
    }

    private static async Task CopyPackageFileAsync(
        string packagePath,
        string destinationPath)
    {
        await using Stream source =
            await FileSystem.OpenAppPackageFileAsync(packagePath);
        await using FileStream destination = File.Create(destinationPath);
        await source.CopyToAsync(destination);
    }

    private static void WriteStandardError(string text)
    {
        if (!string.IsNullOrWhiteSpace(text))
        {
            Debug.WriteLine(text);
        }
    }

    private static string TrimForDiagnostic(string value)
    {
        const int maximumLength = 4000;
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value.Length <= maximumLength
            ? value
            : value[^maximumLength..];
    }

    private sealed record ProcessOutput(
        int ExitCode,
        string StandardOutput,
        string StandardError);
#endif
}
