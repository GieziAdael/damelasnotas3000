using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;

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

        ProcessOutput originalTranscription = await RunTranscriptionAsync(
            pythonExe,
            modelDirectory,
            wavPath,
            outputDirectory);

        Debug.WriteLine("Transcripción Magenta original:");
        Debug.WriteLine(originalTranscription.StandardOutput);
        WriteStandardError(originalTranscription.StandardError);

        if (originalTranscription.ExitCode != 0)
        {
            throw new InvalidOperationException(
                string.IsNullOrWhiteSpace(originalTranscription.StandardError)
                    ? originalTranscription.StandardOutput
                    : originalTranscription.StandardError);
        }

        string? midiPath = Directory
            .GetFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly)
            .FirstOrDefault(path =>
                Path.GetExtension(path).Equals(
                    ".midi", StringComparison.OrdinalIgnoreCase) ||
                Path.GetExtension(path).Equals(
                    ".mid", StringComparison.OrdinalIgnoreCase));

        if (midiPath is null)
        {
            throw new FileNotFoundException(
                $"Magenta terminó, pero no generó un MIDI en {outputDirectory}." +
                $"{Environment.NewLine}{originalTranscription.StandardOutput}" +
                $"{Environment.NewLine}{originalTranscription.StandardError}");
        }

        string originalBackupPath = Path.Combine(
            outputDirectory,
            "grabacion_magenta_original.mid");
        File.Copy(midiPath, originalBackupPath, overwrite: true);

        try
        {
            const double prerollSeconds = 1.0;
            string boundaryScriptPath = Path.Combine(
                outputDirectory,
                "start_boundary_recovery.py");
            await CopyPackageFileAsync(
                "MusicAnalysis/start_boundary_recovery.py",
                boundaryScriptPath);

            ProcessOutput boundaryAnalysis = await RunBoundaryScriptAsync(
                pythonExe,
                boundaryScriptPath,
                outputDirectory,
                "analyze",
                "--audio", wavPath,
                "--midi", midiPath,
                "--output-dir", outputDirectory,
                "--preroll-seconds",
                prerollSeconds.ToString(CultureInfo.InvariantCulture));

            Debug.WriteLine("Análisis del inicio de la grabación:");
            Debug.WriteLine(boundaryAnalysis.StandardOutput);
            WriteStandardError(boundaryAnalysis.StandardError);

            if (boundaryAnalysis.ExitCode != 0)
            {
                return midiPath;
            }

            JsonNode? analysis = ParseMarkedJson(
                boundaryAnalysis.StandardOutput,
                "BOUNDARY_ANALYSIS:");
            bool shouldRetry =
                analysis?["shouldRetry"]?.GetValue<bool>() ?? false;
            if (!shouldRetry)
            {
                return midiPath;
            }

            string paddedAudioPath = Path.Combine(
                outputDirectory,
                "grabacion_con_preroll.wav");
            if (!File.Exists(paddedAudioPath))
            {
                Debug.WriteLine(
                    "No se creó el WAV con preámbulo; se conserva el MIDI original.");
                return midiPath;
            }

            ProcessOutput paddedTranscription = await RunTranscriptionAsync(
                pythonExe,
                modelDirectory,
                paddedAudioPath,
                outputDirectory);

            Debug.WriteLine("Segunda pasada Magenta con preámbulo:");
            Debug.WriteLine(paddedTranscription.StandardOutput);
            WriteStandardError(paddedTranscription.StandardError);

            string? paddedMidiPath = Directory
                .GetFiles(outputDirectory, "grabacion_con_preroll.wav*", SearchOption.TopDirectoryOnly)
                .FirstOrDefault(path =>
                    Path.GetExtension(path).Equals(
                        ".midi", StringComparison.OrdinalIgnoreCase) ||
                    Path.GetExtension(path).Equals(
                        ".mid", StringComparison.OrdinalIgnoreCase));
            if (paddedTranscription.ExitCode != 0 ||
                paddedMidiPath is null)
            {
                Debug.WriteLine(
                    "La segunda pasada no produjo MIDI; se conserva el original.");
                return midiPath;
            }

            string correctedCandidatePath = Path.Combine(
                outputDirectory,
                "grabacion_inicio_revisado.mid");
            ProcessOutput mergeOutput = await RunBoundaryScriptAsync(
                pythonExe,
                boundaryScriptPath,
                outputDirectory,
                "merge",
                "--audio", wavPath,
                "--original-midi", midiPath,
                "--padded-midi", paddedMidiPath,
                "--output-midi", correctedCandidatePath,
                "--output-dir", outputDirectory,
                "--preroll-seconds",
                prerollSeconds.ToString(CultureInfo.InvariantCulture));

            Debug.WriteLine("Revisión de notas del inicio:");
            Debug.WriteLine(mergeOutput.StandardOutput);
            WriteStandardError(mergeOutput.StandardError);

            if (mergeOutput.ExitCode != 0)
            {
                return midiPath;
            }

            JsonNode? recovery = ParseMarkedJson(
                mergeOutput.StandardOutput,
                "BOUNDARY_RECOVERY:");
            int addedNotes =
                recovery?["candidateCount"]?.GetValue<int>() ?? 0;

            return addedNotes > 0 && File.Exists(correctedCandidatePath)
                ? correctedCandidatePath
                : midiPath;
        }
        catch (Exception exception)
        {
            // El rescate es opcional: cualquier fallo mantiene operativo el
            // flujo de transcripción original.
            Debug.WriteLine(
                "No se pudo ejecutar el rescate del inicio; " +
                "se conserva el MIDI Magenta original. " + exception);
            return midiPath;
        }
#else
        throw new PlatformNotSupportedException(
            "La integración local con Magenta está configurada para Windows.");
#endif
    }

#if WINDOWS
    private static async Task<ProcessOutput> RunTranscriptionAsync(
        string pythonExe,
        string modelDirectory,
        string wavPath,
        string workingDirectory)
    {
        var startInfo = CreatePythonStartInfo(
            pythonExe,
            workingDirectory);
        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add(
            "magenta.models.onsets_frames_transcription." +
            "onsets_frames_transcription_transcribe");
        startInfo.ArgumentList.Add($"--model_dir={modelDirectory}");
        startInfo.ArgumentList.Add(wavPath);
        return await RunProcessAsync(startInfo);
    }

    private static async Task<ProcessOutput> RunBoundaryScriptAsync(
        string pythonExe,
        string scriptPath,
        string workingDirectory,
        string mode,
        params string[] arguments)
    {
        var startInfo = CreatePythonStartInfo(
            pythonExe,
            workingDirectory);
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add(mode);
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        return await RunProcessAsync(startInfo);
    }

    private static ProcessStartInfo CreatePythonStartInfo(
        string pythonExe,
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
        return startInfo;
    }

    private static async Task<ProcessOutput> RunProcessAsync(
        ProcessStartInfo startInfo)
    {
        using var process = new Process { StartInfo = startInfo };
        if (!process.Start())
        {
            throw new InvalidOperationException(
                $"No se pudo iniciar el proceso {startInfo.FileName}.");
        }

        Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return new ProcessOutput(
            process.ExitCode,
            await stdoutTask,
            await stderrTask);
    }

    private static JsonNode? ParseMarkedJson(string output, string marker)
    {
        string? jsonLine = output
            .Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault(line => line.StartsWith(marker, StringComparison.Ordinal));

        return jsonLine is null
            ? null
            : JsonNode.Parse(jsonLine[marker.Length..]);
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

    private sealed record ProcessOutput(
        int ExitCode,
        string StandardOutput,
        string StandardError);

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
