using System.Diagnostics;

namespace Damelasnotas3000.Services;

public class MidiToMusicXmlService
{
    public async Task<string> ConvertAsync(
        string midiFilePath,
        string outputDirectory)
    {
#if WINDOWS
        if (!File.Exists(midiFilePath))
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

        string converterPackagesDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreConverterPackages");

        string musicXmlPath = Path.Combine(
            outputDirectory,
            "grabacion.musicxml");

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.Environment["PYTHONPATH"] = converterPackagesDirectory;
        startInfo.Environment["MIDI_INPUT"] = midiFilePath;
        startInfo.Environment["XML_OUTPUT"] = musicXmlPath;

        startInfo.ArgumentList.Add("-c");
        startInfo.ArgumentList.Add(
            "import os; " +
            "from music21 import converter; " +
            "score = converter.parse(os.environ['MIDI_INPUT']); " +
            "score.write('musicxml', fp=os.environ['XML_OUTPUT']); " +
            "print('MusicXML generado:', os.environ['XML_OUTPUT'])");

        using var process = new Process { StartInfo = startInfo };
        process.Start();

        Task<string> stdoutTask =
            process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask =
            process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (process.ExitCode != 0 || !File.Exists(musicXmlPath))
        {
            throw new InvalidOperationException(
                "No se pudo convertir el MIDI a MusicXML." +
                $"{Environment.NewLine}{stderr}{Environment.NewLine}{stdout}");
        }

        return musicXmlPath;
#else
        throw new PlatformNotSupportedException(
            "La conversión local está configurada para Windows.");
#endif
    }
}