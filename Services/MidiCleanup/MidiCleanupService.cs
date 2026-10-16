using System.Diagnostics;
using System.Text.Json;
using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services.MidiCleanup;

public sealed class MidiCleanupService
{
    public async Task<MidiCleanupResult> AnalyzeAsync(
        string audioFilePath,
        string midiFilePath,
        string outputDirectory)
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
                "No se encontró el MIDI para evaluar.",
                midiFilePath);
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

        string scriptPath = Path.Combine(
            outputDirectory,
            "clean_midi.py");

        await using (Stream source =
            await FileSystem.OpenAppPackageFileAsync(
                "MidiCleanup/clean_midi.py"))
        await using (FileStream destination =
            File.Create(scriptPath))
        {
            await source.CopyToAsync(destination);
        }

        string reportPath = Path.Combine(
            outputDirectory,
            "midi_cleanup_diagnostic.json");

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.Environment["AUDIO_INPUT"] = audioFilePath;
        startInfo.Environment["MIDI_INPUT"] = midiFilePath;

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

        Debug.WriteLine("Diagnóstico MIDI:");
        Debug.WriteLine(stdout);

        if (!string.IsNullOrWhiteSpace(stderr))
        {
            Debug.WriteLine(stderr);
        }

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "Falló el diagnóstico de eventos MIDI." +
                $"{Environment.NewLine}{stderr}" +
                $"{Environment.NewLine}{stdout}");
        }

        string? jsonLine = stdout
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
                "El analizador MIDI no devolvió el JSON esperado." +
                $"{Environment.NewLine}{stdout}");
        }

        string json = jsonLine["CLEANUP_JSON:".Length..];

        MidiCleanupResult? result =
            JsonSerializer.Deserialize<MidiCleanupResult>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

        if (result is null)
        {
            throw new InvalidOperationException(
                "No se pudo leer el diagnóstico de eventos MIDI.");
        }

        MidiCleanupResult resultWithPath =
            result with { ReportPath = reportPath };

        string prettyJson = JsonSerializer.Serialize(
            resultWithPath,
            new JsonSerializerOptions
            {
                WriteIndented = true
            });

        await File.WriteAllTextAsync(
            reportPath,
            prettyJson);

        return resultWithPath;
#else
        throw new PlatformNotSupportedException(
            "El diagnóstico local de eventos está configurado para Windows.");
#endif
    }
}