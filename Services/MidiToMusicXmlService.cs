using System.Diagnostics;
using System.Globalization;
using Damelasnotas3000.Models;

namespace Damelasnotas3000.Services;

public class MidiToMusicXmlService
{
    public async Task<string> ConvertAsync(
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
        }

        double? tempoBpm = tempoBpmOverride ?? analysis.SelectedTempoBpm;
        if (tempoBpm is double bpm && double.IsFinite(bpm) && bpm > 0)
        {
            startInfo.Environment["TEMPO_BPM"] =
                bpm.ToString("0.###", CultureInfo.InvariantCulture);

            // Si se introduce un BPM manual sin unidad, se interpreta como
            // negra, que es la convención habitual de las indicaciones BPM.
            string tempoUnit = tempoUnitOverride ??
                (tempoBpmOverride.HasValue
                    ? "quarter"
                    : analysis.SelectedTempoUnit ?? "quarter");

            startInfo.Environment["TEMPO_UNIT"] = tempoUnit;
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

        Debug.WriteLine(stdout);
        Debug.WriteLine(stderr);

        if (process.ExitCode != 0 || !File.Exists(musicXmlPath))
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
