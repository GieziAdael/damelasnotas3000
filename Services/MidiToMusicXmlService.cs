using System.Diagnostics;
using System.Globalization;
using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services;

public class MidiToMusicXmlService
{
    public async Task<string> ConvertAsync(
        string midiFilePath,
        string outputDirectory,
        MusicAnalysisResult analysis)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(midiFilePath) ||
            !File.Exists(midiFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el archivo MIDI.",
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

        string converterPackagesDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreConverterPackages");

        string musicXmlPath = Path.Combine(
            outputDirectory,
            "grabacion.musicxml");

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

        startInfo.Environment["MIDI_INPUT"] =
            midiFilePath;

        startInfo.Environment["XML_OUTPUT"] =
            musicXmlPath;

        // Solo se envía el compás cuando el análisis lo seleccionó.
        if (analysis.TimeSignatureNumerator is int numerator &&
            analysis.TimeSignatureDenominator is int denominator)
        {
            startInfo.Environment["METER_NUMERATOR"] =
                numerator.ToString(CultureInfo.InvariantCulture);

            startInfo.Environment["METER_DENOMINATOR"] =
                denominator.ToString(CultureInfo.InvariantCulture);
        }

        // Usa el tempo de Librosa si está disponible; si no, el de Madmom.
        double? tempoBpm =
            analysis.TempoBpm ?? analysis.MadmomTempoBpm;

        if (tempoBpm.HasValue &&
            double.IsFinite(tempoBpm.Value) &&
            tempoBpm.Value > 0)
        {
            startInfo.Environment["TEMPO_BPM"] =
                tempoBpm.Value.ToString(
                    CultureInfo.InvariantCulture);
        }

        // Se ejecuta el archivo .py directamente; no se usa "-c".
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

        if (process.ExitCode != 0 ||
            !File.Exists(musicXmlPath))
        {
            throw new InvalidOperationException(
                "No se pudo convertir el MIDI a MusicXML." +
                $"{Environment.NewLine}{stderr}" +
                $"{Environment.NewLine}{stdout}");
        }

        return musicXmlPath;
#else
        throw new PlatformNotSupportedException(
            "La conversión local está configurada para Windows.");
#endif
    }
}