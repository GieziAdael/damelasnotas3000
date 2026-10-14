
using Plugin.Maui.Audio;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace Damelasnotas3000;

public partial class MainPage : ContentPage
{
    private readonly IAudioRecorder _recorder;
    private string? _audioFilePath;
    private bool _isRecording;
    private string? _midiFilePath;
#if WINDOWS
    private NAudio.Midi.MidiOut? _midiOut;
#endif
    private System.Threading.CancellationTokenSource? _playbackCts;
    private bool _isPlaying;

    public MainPage()
    {
        InitializeComponent();

        _recorder = AudioManager.Current.CreateRecorder();

        DebugStatus.Text = "MainPage creada correctamente";
    }

    private async void OnRecordClicked(
        object sender, EventArgs e)
    {
        try
        {
#if WINDOWS
// Prueba temporal: omitir la solicitud de permiso de MAUI en Windows.
#else
            var permission = await Permissions.RequestAsync<
                Permissions.Microphone>();

            if (permission != PermissionStatus.Granted)
            {
                await DisplayAlertAsync(
                    "Permiso requerido",
                    "Debes permitir el acceso al micrófono.",
                    "Aceptar");
                return;
            }
#endif

            DeleteAudioFile();

            await _recorder.StartAsync();
            _isRecording = true;

            RecordingStatus.Text = "Grabando audio...";
            RecordButton.IsEnabled = false;
            StopButton.IsEnabled = true;
            DiscardButton.IsEnabled = true;
            ConvertButton.IsEnabled = false;

            AudioFileStatus.Text = "Grabación en curso";
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync(
                "Error",
                ex.Message,
                "Aceptar");
        }
    }

    private async void OnStopClicked(
        object sender, EventArgs e)
    {
        if (!_isRecording)
            return;

        try
        {
            var audioSource = await _recorder.StopAsync();

            if (audioSource is not FileAudioSource fileSource)
            {
                RecordingStatus.Text =
                    "No se pudo obtener el archivo de audio.";
                return;
            }

            _audioFilePath = fileSource.GetFilePath();
            var fileInfo = new FileInfo(_audioFilePath);

            
            _isRecording = false;

            RecordingStatus.Text = "Grabación finalizada";
            AudioFileStatus.Text =
                $"Audio guardado: {Path.GetFileName(_audioFilePath)}";

            RecordButton.IsEnabled = true;
            StopButton.IsEnabled = false;
            DiscardButton.IsEnabled = true;
            ConvertButton.IsEnabled = true;


        }
        catch (Exception ex)
        {
            await DisplayAlertAsync(
                "Error",
                ex.Message,
                "Aceptar");
        }
    }

    private async void OnDiscardClicked(
        object sender, EventArgs e)
    {
        if (_isRecording)
        {
            await _recorder.StopAsync();
            _isRecording = false;
        }

        DeleteAudioFile();

        RecordingStatus.Text = "Listo para grabar";
        AudioFileStatus.Text = "No hay audio grabado";

        RecordButton.IsEnabled = true;
        StopButton.IsEnabled = false;
        DiscardButton.IsEnabled = false;
        ConvertButton.IsEnabled = false;
    }

    private void DeleteAudioFile()
    {
        if (!string.IsNullOrWhiteSpace(_audioFilePath)
            && File.Exists(_audioFilePath))
        {
            File.Delete(_audioFilePath);
        }

        _audioFilePath = null;
    }



    private async void OnConvertClicked(object sender, EventArgs e)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(_audioFilePath) ||
            !File.Exists(_audioFilePath))
        {
            await DisplayAlertAsync(
                "Error",
                "No se encontró el archivo de audio. Graba primero.",
                "Aceptar");
            return;
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
        await DisplayAlertAsync(
            "Error",
            $"No se encontró Python o el modelo de Magenta.\n\n" +
            $"Python esperado:\n{pythonExe}\n\n" +
            $"Modelo esperado:\n{modelDirectory}",
            "Aceptar");
        return;
    }

    ConvertButton.IsEnabled = false;
    RecordButton.IsEnabled = false;
    StopButton.IsEnabled = false;
    DiscardButton.IsEnabled = false;

    try
    {
        RecordingStatus.Text = "Transcribiendo con Magenta...";

        string outputDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "MidiOutput",
            DateTime.Now.ToString("yyyyMMdd_HHmmss"));

        Directory.CreateDirectory(outputDirectory);

        // La grabación tiene extensión .ksb,
        // pero comprobamos que su contenido ya es WAV.
        string wavPath = Path.Combine(
            outputDirectory,
            "grabacion.wav");

        File.Copy(_audioFilePath, wavPath, true);

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
            "magenta.models.onsets_frames_transcription.onsets_frames_transcription_transcribe");

        startInfo.ArgumentList.Add(
            $"--model_dir={modelDirectory}");

        startInfo.ArgumentList.Add(wavPath);

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

        if (process.ExitCode != 0)
        {
            RecordingStatus.Text = "Error de transcripción.";

            await DisplayAlertAsync(
                "Error de Magenta",
                string.IsNullOrWhiteSpace(stderr)
                    ? stdout
                    : stderr,
                "Aceptar");

            return;
        }

        // Magenta suele generar un archivo .midi.
        // También comprobamos la extensión .mid.
        string? midiPath = Directory
            .GetFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly)
            .FirstOrDefault(path =>
                Path.GetExtension(path).Equals(
                    ".midi",
                    StringComparison.OrdinalIgnoreCase) ||
                Path.GetExtension(path).Equals(
                    ".mid",
                    StringComparison.OrdinalIgnoreCase));

        if (midiPath is null)
        {
            RecordingStatus.Text =
                "Magenta terminó, pero no se encontró el MIDI.";

            await DisplayAlertAsync(
                "Resultado no encontrado",
                $"Magenta terminó sin error, pero no encontramos el MIDI.\n\n" +
                $"Carpeta de salida:\n{outputDirectory}\n\n" +
                $"Salida del proceso:\n{stdout}\n{stderr}",
                "Aceptar");

            return;
        }

        _midiFilePath = midiPath;

        // Convertir el MIDI recién generado a MusicXML.
        RecordingStatus.Text = "Convirtiendo MIDI a MusicXML...";

        string converterPackagesDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreConverterPackages");

        string musicXmlPath = Path.Combine(
            outputDirectory,
            "grabacion.musicxml");

        var convertInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = outputDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        convertInfo.Environment["PYTHONPATH"] =
            converterPackagesDirectory;

        convertInfo.Environment["MIDI_INPUT"] = midiPath;
        convertInfo.Environment["XML_OUTPUT"] = musicXmlPath;

        convertInfo.ArgumentList.Add("-c");
        convertInfo.ArgumentList.Add(
            "import os; " +
            "from music21 import converter; " +
            "score = converter.parse(os.environ['MIDI_INPUT']); " +
            "score.write('musicxml', fp=os.environ['XML_OUTPUT']); " +
            "print('MusicXML generado:', os.environ['XML_OUTPUT'])");

        using (var convertProcess = new Process
        {
            StartInfo = convertInfo
        })
        {
            convertProcess.Start();

            Task<string> convertStdoutTask =
                convertProcess.StandardOutput.ReadToEndAsync();

            Task<string> convertStderrTask =
                convertProcess.StandardError.ReadToEndAsync();

            await convertProcess.WaitForExitAsync();

            string convertStdout = await convertStdoutTask;
            string convertStderr = await convertStderrTask;

            if (convertProcess.ExitCode != 0 ||
                !File.Exists(musicXmlPath))
            {
                throw new Exception(
                    "No se pudo convertir el MIDI a MusicXML.\n\n" +
                    convertStderr + "\n" + convertStdout);
            }
        }

        string scoreViewerDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreViewer");

        Directory.CreateDirectory(scoreViewerDirectory);

        string viewerMusicXmlPath = Path.Combine(
            scoreViewerDirectory,
            "grabacion.musicxml");

        File.Copy(musicXmlPath, viewerMusicXmlPath, true);

        // Cargar la partitura generada en el WebView.
        await LoadScoreViewerAsync("grabacion.musicxml");

        RecordingStatus.Text = "Transcripción completada.";

        await DisplayAlertAsync(
            "Conversión completada",
            $"El archivo MIDI se generó correctamente:\n\n{midiPath}",
            "Aceptar");
    }
    catch (Exception ex)
    {
        RecordingStatus.Text = "Error de conversión.";

        await DisplayAlertAsync(
            "Error",
            ex.Message,
            "Aceptar");
    }
    finally
    {
        RecordButton.IsEnabled = true;
        StopButton.IsEnabled = false;
        DiscardButton.IsEnabled = false;
        ConvertButton.IsEnabled = false;

        PlayButton.IsEnabled =
            !string.IsNullOrWhiteSpace(_midiFilePath) &&
            File.Exists(_midiFilePath);
    }
#else
        await DisplayAlertAsync(
            "No disponible",
            "La integración local con Magenta está configurada para Windows.",
            "Aceptar");
#endif
    }


    private async Task<string> PrepareMagentaDirectoryAsync()
    {
        string sourceDirectory = Path.Combine(
            AppContext.BaseDirectory,
            "Magenta");

        string destinationDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "Magenta");

        string destinationPython = Path.Combine(
            destinationDirectory,
            "Runtime",
            "Scripts",
            "python.exe");

        string destinationModel = Path.Combine(
            destinationDirectory,
            "Models",
            "checkpoint",
            "train");

        string destinationPythonBase = Path.Combine(
            destinationDirectory,
            "PythonBase");

        if (!File.Exists(Path.Combine(
                sourceDirectory,
                "Runtime",
                "Scripts",
                "python.exe")) ||
            !Directory.Exists(Path.Combine(
                sourceDirectory,
                "Models",
                "checkpoint",
                "train")) ||
            !Directory.Exists(Path.Combine(
                sourceDirectory,
                "PythonBase")))
        {
            throw new DirectoryNotFoundException(
                $"No se encontró la distribución de Magenta en:\n{sourceDirectory}");
        }

        bool alreadyPrepared =
            File.Exists(destinationPython) &&
            Directory.Exists(destinationModel) &&
            Directory.Exists(destinationPythonBase);

        if (!alreadyPrepared)
        {
            if (Directory.Exists(destinationDirectory))
            {
                Directory.Delete(destinationDirectory, true);
            }

            CopyDirectory(sourceDirectory, destinationDirectory);
        }

        string configPath = Path.Combine(
            destinationDirectory,
            "Runtime",
            "pyvenv.cfg");

        string config = await File.ReadAllTextAsync(configPath);

        string homeLine = $"home = {destinationPythonBase}";

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
                lines[i] = homeLine;
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

            File.Copy(file, destinationFile, true);
        }

        foreach (string directory in Directory.GetDirectories(sourceDirectory))
        {
            string destinationSubdirectory = Path.Combine(
                destinationDirectory,
                Path.GetFileName(directory));

            CopyDirectory(directory, destinationSubdirectory);
        }
    }



    private static string? FindBasicPitchExecutable()
    {
        // Busca la carpeta del proyecto a partir de la ubicación
        // de ejecución de la aplicación.
        DirectoryInfo? directory =
            new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null)
        {
            string candidate = Path.Combine(
                directory.FullName,
                ".venv",
                "Scripts",
                "basic-pitch.exe");

            if (File.Exists(candidate))
                return candidate;

            directory = directory.Parent;
        }

        return null;
    }

    //Visor

    private async Task<string> PrepareScoreViewerAsync()
    {
        string destinationDirectory = Path.Combine(
            FileSystem.AppDataDirectory,
            "ScoreViewer"
        );

        Directory.CreateDirectory(destinationDirectory);

        string[] files =
        {
        "index.html",
        "opensheetmusicdisplay.min.js",
        "prueba.musicxml"
    };

        foreach (string file in files)
        {
            string destinationPath = Path.Combine(
                destinationDirectory,
                file
            );

            // Conserva los archivos existentes para evitar
            // copiarlos de nuevo en cada apertura.
            if (File.Exists(destinationPath))
                continue;

            string assetPath = $"ScoreViewer/{file}";

            await using Stream source =
                await FileSystem.OpenAppPackageFileAsync(assetPath);

            await using FileStream destination = File.Create(destinationPath);

            await source.CopyToAsync(destination);
        }

        return Path.Combine(destinationDirectory, "index.html");
    }


    private async Task LoadScoreViewerAsync(
      string fileName = "prueba.musicxml")
    {
        try
        {
            DebugStatus.Text = "Preparando archivos del visor...";

            string htmlPath = await PrepareScoreViewerAsync();
            string directory = Path.GetDirectoryName(htmlPath)!;

            string xmlPath = Path.Combine(directory, fileName);
            string jsPath = Path.Combine(
                directory,
                "opensheetmusicdisplay.min.js");

            if (!File.Exists(htmlPath))
                throw new FileNotFoundException("No existe index.html.", htmlPath);

            if (!File.Exists(jsPath))
                throw new FileNotFoundException("No existe OSMD.", jsPath);

            if (!File.Exists(xmlPath))
                throw new FileNotFoundException(
                    $"No existe el archivo MusicXML solicitado: {fileName}",
                    xmlPath);

            long xmlSize = new FileInfo(xmlPath).Length;

            DebugStatus.Text =
                $"Archivos encontrados. MusicXML: {xmlSize} bytes. Cargando visor...";


            string viewerUrl =
                $"https://damelasnotas3000.example/index.html" +
                $"?file={Uri.EscapeDataString(fileName)}" +
                $"&t={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}";

            DebugStatus.Text = $"Cargando: {fileName}\n{xmlSize} bytes";

            ScoreWebView.Source = new UrlWebViewSource
            {
                Url = viewerUrl
            };

        }
        catch (Exception ex)
        {
            DebugStatus.Text = $"Error: {ex.Message}";
        }
    }



    protected override void OnAppearing()
    {
        base.OnAppearing();

        DebugStatus.Text = "OnAppearing ejecutado";

        // Solo carga la partitura de ejemplo al abrir la página.
        // La conversión podrá cargar grabacion.musicxml después.
        if (ScoreWebView.Source is null)
        {
            _ = LoadScoreViewerAsync("prueba.musicxml");
        }
    }




    private void ScoreWebView_Navigated(
        object? sender,
        WebNavigatedEventArgs e)
    {
        DebugStatus.Text =
            $"Navegación del visor: {e.Result}\n{e.Url}";
    }


    private async void OnPlayClicked(object sender, EventArgs e)
    {
        await PlayMidiAsync();
    }

    private void OnStopPlaybackClicked(object sender, EventArgs e)
    {
        _playbackCts?.Cancel();
    }

    private async Task PlayMidiAsync()
    {
#if WINDOWS
    if (string.IsNullOrWhiteSpace(_midiFilePath) ||
        !File.Exists(_midiFilePath))
    {
        await DisplayAlertAsync(
            "MIDI",
            "Primero graba y convierte el audio.",
            "Aceptar");
        return;
    }

    if (_isPlaying)
        return;

    try
    {
        _isPlaying = true;
        PlayButton.IsEnabled = false;
        StopPlaybackButton.IsEnabled = true;

        _playbackCts = new CancellationTokenSource();
        CancellationToken token = _playbackCts.Token;

        var midiFile = new NAudio.Midi.MidiFile(
            _midiFilePath,
            false);

        _midiOut = new NAudio.Midi.MidiOut(0);

        var events = midiFile.Events
            .SelectMany(track => track)
            .Where(e => e is NAudio.Midi.NoteOnEvent ||
                        e is NAudio.Midi.NoteEvent ||
                        e is NAudio.Midi.TempoEvent)
            .OrderBy(e => e.AbsoluteTime)
            .ToList();

        long previousTime = 0;
        int tempo = 500000;
        int ticksPerQuarter = midiFile.DeltaTicksPerQuarterNote;

        foreach (var midiEvent in events)
        {
            token.ThrowIfCancellationRequested();

            long deltaTicks = midiEvent.AbsoluteTime - previousTime;

            if (deltaTicks > 0)
            {
                double milliseconds =
                    deltaTicks * tempo /
                    (double)ticksPerQuarter / 1000.0;

                await Task.Delay(
                    TimeSpan.FromMilliseconds(milliseconds),
                    token);
            }

            previousTime = midiEvent.AbsoluteTime;

            if (midiEvent is NAudio.Midi.TempoEvent tempoEvent)
            {
                tempo = tempoEvent.MicrosecondsPerQuarterNote;
            }
            else if (midiEvent is NAudio.Midi.NoteOnEvent noteOn)
            {
                _midiOut.Send(noteOn.GetAsShortMessage());
            }
            else if (midiEvent is NAudio.Midi.NoteEvent noteEvent)
            {
                _midiOut.Send(noteEvent.GetAsShortMessage());
            }
        }
    }
    catch (OperationCanceledException)
    {
        // La reproducción fue detenida por el usuario.
    }
    catch (Exception ex)
    {
        await DisplayAlertAsync(
            "Error de reproducción",
            ex.Message,
            "Aceptar");
    }
    finally
    {
        _midiOut?.Dispose();
        _midiOut = null;

        _playbackCts?.Dispose();
        _playbackCts = null;

        _isPlaying = false;

        PlayButton.IsEnabled =
            !string.IsNullOrWhiteSpace(_midiFilePath) &&
            File.Exists(_midiFilePath);

        StopPlaybackButton.IsEnabled = false;
    }
#endif
    }

    private async Task CheckMidiDevicesAsync()
    {
#if WINDOWS
        await DisplayAlertAsync(
            "Prueba MIDI",
            "El código de Windows se está ejecutando.",
            "Aceptar");

        int deviceCount = NAudio.Midi.MidiOut.NumberOfDevices;

        if (deviceCount == 0)
        {
            await DisplayAlertAsync(
                "MIDI",
                "No se encontraron dispositivos de salida MIDI.",
                "Aceptar");
            return;
        }

        var devices = new System.Text.StringBuilder();

        for (int i = 0; i < deviceCount; i++)
        {
            var device = NAudio.Midi.MidiOut.DeviceInfo(i);
            devices.AppendLine($"{i}: {device.ProductName}");
        }

        await DisplayAlertAsync(
            "Dispositivos MIDI",
            devices.ToString(),
            "Aceptar");
#else
        await DisplayAlertAsync(
            "Plataforma",
            "Este código solo está habilitado para Windows.",
            "Aceptar");
    #endif
    }

}

