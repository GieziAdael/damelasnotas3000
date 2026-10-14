using System.Diagnostics;

namespace Damelasnotas3000.Services;

public class MagentaService
{
    public async Task<string> TranscribeAsync(
        string audioFilePath,
        string outputDirectory)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(audioFilePath) ||
            !File.Exists(audioFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el archivo de audio.",
                audioFilePath);
        }

        string distributionDirectory =
            await PrepareMagentaDirectoryAsync();

        string pythonExe = Path.Combine(
            distributionDirectory,
            "Runtime",
            "Scripts",
            "python.exe");

        string modelDirectory = Path.Combine(
            distributionDirectory,
            "Models",
            "checkpoint",
            "train");

        if (!File.Exists(pythonExe) ||
            !Directory.Exists(modelDirectory))
        {
            throw new FileNotFoundException(
                $"No se encontró Python o el modelo de Magenta." +
                $"{Environment.NewLine}Python: {pythonExe}" +
                $"{Environment.NewLine}Modelo: {modelDirectory}");
        }

        Directory.CreateDirectory(outputDirectory);

        string wavPath = Path.Combine(outputDirectory, "grabacion.wav");
        File.Copy(audioFilePath, wavPath, overwrite: true);

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add(
            "magenta.models.onsets_frames_transcription." +
            "onsets_frames_transcription_transcribe");
        startInfo.ArgumentList.Add($"--model_dir={modelDirectory}");
        startInfo.ArgumentList.Add(wavPath);

        using var process = new Process { StartInfo = startInfo };
        process.Start();

        Task<string> stdoutTask =
            process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask =
            process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                string.IsNullOrWhiteSpace(stderr) ? stdout : stderr);
        }

        string? midiPath = Directory
            .GetFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly)
            .FirstOrDefault(path =>
                Path.GetExtension(path).Equals(
                    ".midi", StringComparison.OrdinalIgnoreCase) ||
                Path.GetExtension(path).Equals(
                    ".mid", StringComparison.OrdinalIgnoreCase));

        return midiPath ?? throw new FileNotFoundException(
            $"Magenta terminó, pero no generó un MIDI en {outputDirectory}." +
            $"{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
#else
        throw new PlatformNotSupportedException(
            "La integración local con Magenta está configurada para Windows.");
#endif
    }

#if WINDOWS
    private static async Task<string> PrepareMagentaDirectoryAsync()
    {
        string sourceDirectory = Path.Combine(
            AppContext.BaseDirectory,
            "Magenta");

        string destinationDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "Magenta");

        string sourcePython = Path.Combine(
            sourceDirectory, "Runtime", "Scripts", "python.exe");
        string sourceModel = Path.Combine(
            sourceDirectory, "Models", "checkpoint", "train");
        string sourcePythonBase = Path.Combine(
            sourceDirectory, "PythonBase");

        if (!File.Exists(sourcePython) ||
            !Directory.Exists(sourceModel) ||
            !Directory.Exists(sourcePythonBase))
        {
            throw new DirectoryNotFoundException(
                $"La distribución de Magenta está incompleta: {sourceDirectory}");
        }

        string destinationPython = Path.Combine(
            destinationDirectory, "Runtime", "Scripts", "python.exe");
        string destinationModel = Path.Combine(
            destinationDirectory, "Models", "checkpoint", "train");
        string destinationPythonBase = Path.Combine(
            destinationDirectory, "PythonBase");

        bool alreadyPrepared =
            File.Exists(destinationPython) &&
            Directory.Exists(destinationModel) &&
            Directory.Exists(destinationPythonBase);

        if (!alreadyPrepared)
        {
            // No se elimina aquí una carpeta existente. Se copian los archivos
            // faltantes o actualizados sobre el destino.
            CopyDirectory(sourceDirectory, destinationDirectory);
        }

        string configPath = Path.Combine(
            destinationDirectory, "Runtime", "pyvenv.cfg");

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                "No se encontró pyvenv.cfg.", configPath);
        }

        string config = await File.ReadAllTextAsync(configPath);
        var lines = config.Split(
            new[] { "\r\n", "\n" },
            StringSplitOptions.None);

        bool homeUpdated = false;

        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].StartsWith(
                    "home =",
                    StringComparison.OrdinalIgnoreCase))
            {
                lines[i] = $"home = {destinationPythonBase}";
                homeUpdated = true;
                break;
            }
        }

        if (!homeUpdated)
        {
            throw new InvalidOperationException(
                "No se encontró la línea 'home' en pyvenv.cfg.");
        }

        await File.WriteAllTextAsync(
            configPath,
            string.Join(Environment.NewLine, lines));

        return destinationDirectory;
    }

    private static void CopyDirectory(
        string sourceDirectory,
        string destinationDirectory)
    {
        Directory.CreateDirectory(destinationDirectory);

        foreach (string file in Directory.GetFiles(sourceDirectory))
        {
            string destinationFile = Path.Combine(
                destinationDirectory,
                Path.GetFileName(file));

            File.Copy(file, destinationFile, overwrite: true);
        }

        foreach (string directory in Directory.GetDirectories(sourceDirectory))
        {
            string destinationSubdirectory = Path.Combine(
                destinationDirectory,
                Path.GetFileName(directory));

            CopyDirectory(directory, destinationSubdirectory);
        }
    }
#endif
}