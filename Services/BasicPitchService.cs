using System.Diagnostics;

namespace Damelasnotas3000.Services;

/// <summary>
/// Ejecuta Basic Pitch desde el entorno Python local y guarda su MIDI original.
/// </summary>
public sealed class BasicPitchService
{
    private static readonly TimeSpan TranscriptionTimeout =
        TimeSpan.FromMinutes(5);

    public async Task<string> TranscribeAsync(
        string audioFilePath,
        string outputDirectory)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(audioFilePath) ||
            !File.Exists(audioFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el archivo de audio para Basic Pitch.",
                audioFilePath);
        }

        string pythonExe = FindBasicPitchPythonExecutable()
            ?? throw new FileNotFoundException(
                "No se encontró Python con Basic Pitch. Configura " +
                "BASIC_PITCH_PYTHON o coloca el entorno en .venv\\Scripts\\python.exe.");

        Directory.CreateDirectory(outputDirectory);

        string scriptPath = Path.Combine(
            outputDirectory,
            "compare_transcriptions.py");
        await CopyPackageFileAsync(
            "MusicAnalysis/compare_transcriptions.py",
            scriptPath);

        string midiPath = Path.Combine(
            outputDirectory,
            "grabacion_basic_pitch.mid");
        if (File.Exists(midiPath))
        {
            File.Delete(midiPath);
        }

        var startInfo = CreatePythonStartInfo(
            pythonExe,
            scriptPath,
            outputDirectory);
        startInfo.Environment["AUDIO_INPUT"] = audioFilePath;
        startInfo.Environment["COMPARISON_OUTPUT_DIRECTORY"] = outputDirectory;
        startInfo.Environment["BASIC_PITCH_MIDI_OUTPUT"] = midiPath;
        startInfo.Environment["BASIC_PITCH_MINIMUM_NOTE_LENGTH_MS"] = "45.0";
        startInfo.Environment["BASIC_PITCH_ONLY"] = "1";
        startInfo.Environment.Remove("MAGENTA_MIDI_INPUT");
        startInfo.Environment.Remove("BASIC_PITCH_MIDI_INPUT");

        ProcessOutput output = await RunProcessAsync(
            startInfo,
            TranscriptionTimeout);

        await AnalysisDebugLog.AppendProcessOutputAsync(
            outputDirectory,
            "BASIC PITCH: TRANSCRIPCIÓN",
            output.ExitCode,
            output.StandardOutput,
            output.StandardError);

        Debug.WriteLine("Transcripción Basic Pitch:");
        Debug.WriteLine(output.StandardOutput);
        if (!string.IsNullOrWhiteSpace(output.StandardError))
        {
            Debug.WriteLine(output.StandardError);
        }

        if (output.ExitCode != 0)
        {
            throw new InvalidOperationException(
                "Basic Pitch no pudo transcribir el audio." +
                $"{Environment.NewLine}{output.StandardError}" +
                $"{Environment.NewLine}{output.StandardOutput}");
        }

        if (!File.Exists(midiPath) || new FileInfo(midiPath).Length == 0)
        {
            throw new InvalidOperationException(
                "Basic Pitch terminó sin generar un MIDI válido." +
                $"{Environment.NewLine}{output.StandardOutput}");
        }

        return midiPath;
#else
        throw new PlatformNotSupportedException(
            "La integración local con Basic Pitch está configurada para Windows.");
#endif
    }

#if WINDOWS
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
        TimeSpan timeout)
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

        Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask = process.StandardError.ReadToEndAsync();
        Task waitTask = process.WaitForExitAsync();
        Task completedTask = await Task.WhenAny(waitTask, Task.Delay(timeout));
        bool timedOut = completedTask != waitTask;

        if (timedOut)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (InvalidOperationException)
            {
                // El proceso pudo terminar justo al vencer el límite.
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
                $"La transcripción superó el límite de {timeout.TotalMinutes:F0} minutos.");
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

    private sealed record ProcessOutput(
        int ExitCode,
        string StandardOutput,
        string StandardError);
#endif
}
