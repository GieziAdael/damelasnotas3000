using System.Diagnostics;
using System.Text.Json;
using Damelasnotas3000.Models;

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

        string scriptPath = Path.Combine(
            outputDirectory,
            "meter_analysis.py");

        await using (Stream source =
            await FileSystem.OpenAppPackageFileAsync(
                "MusicAnalysis/meter_analysis.py"))
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

        startInfo.Environment["AUDIO_INPUT"] = analysisWavPath;
        startInfo.Environment["MIDI_INPUT"] = midiPath;

        // Al pasar la ruta del archivo, no se agrega "-c".
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

        return new MusicAnalysisResult(
            result.TempoBpm,
            result.BeatCount,
            result.TimeSignatureNumerator,
            result.TimeSignatureDenominator,
            null,
            null,
            result.MadmomTempoBpm,
            result.TempoCandidatesBpm,
            result.MeterCandidates,
            result.MeterConfidence);
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
        double? MeterConfidence);
}