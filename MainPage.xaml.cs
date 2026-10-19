using System.Globalization;
using Damelasnotas3000.Models;
using Damelasnotas3000.Services;
using Damelasnotas3000.Services.MidiCleanup;
using Damelasnotas3000.Services.MusicAnalysis;
using Plugin.Maui.Audio;

namespace Damelasnotas3000;

public partial class MainPage : ContentPage
{
    private readonly IAudioRecorder _recorder;
    private string? _audioFilePath;
    private bool _isRecording;
    private string? _midiFilePath;
    private readonly AudioRecordingService _audioRecordingService;
    private readonly MagentaService _magentaService;
    private readonly MidiToMusicXmlService _musicXmlService;
    private readonly MidiPlaybackService _midiPlaybackService = new();
    private readonly MusicAnalysisService _musicAnalysisService;
    private readonly MidiCleanupService _midiCleanupService;

    public MainPage()
    {
        InitializeComponent();

        _recorder = AudioManager.Current.CreateRecorder();
        _audioRecordingService = new AudioRecordingService(_recorder);
        _magentaService = new MagentaService();
        _musicXmlService = new MidiToMusicXmlService();
        _musicAnalysisService = new MusicAnalysisService();
        _midiCleanupService = new MidiCleanupService();
    }

    private (int? Numerator, int? Denominator) ReadMeterOverride()
    {
        string? value = MeterOverridePicker.SelectedItem?.ToString();
        if (string.IsNullOrWhiteSpace(value) || value == "Automático")
            return (null, null);

        string[] parts = value.Split('/');
        if (parts.Length != 2 ||
            !int.TryParse(parts[0], NumberStyles.Integer,
                CultureInfo.InvariantCulture, out int numerator) ||
            !int.TryParse(parts[1], NumberStyles.Integer,
                CultureInfo.InvariantCulture, out int denominator))
        {
            throw new FormatException("El compás seleccionado no es válido.");
        }

        return (numerator, denominator);
    }

    private double? ReadTempoOverride()
    {
        string? value = TempoOverrideEntry.Text?.Trim();
        if (string.IsNullOrWhiteSpace(value))
            return null;

        bool parsed = double.TryParse(
            value,
            NumberStyles.Float,
            CultureInfo.CurrentCulture,
            out double tempoBpm);
        if (!parsed)
        {
            parsed = double.TryParse(
                value,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out tempoBpm);
        }

        if (!parsed || !double.IsFinite(tempoBpm) || tempoBpm <= 0)
            throw new FormatException("Escribe un tempo válido mayor que cero.");

        return tempoBpm;
    }

    private string? ReadTempoUnitOverride()
    {
        return TempoUnitPicker.SelectedItem?.ToString() switch
        {
            "Negra" => "quarter",
            "Corchea" => "eighth",
            "Negra con puntillo" => "dotted_quarter",
            "Blanca" => "half",
            _ => null
        };
    }

    /// <summary>
    /// Inicia la grabación de audio cuando el usuario pulsa el botón correspondiente.
    /// </summary>
    /// <param name="sender">El control que generó el evento.</param>
    /// <param name="e">Los datos asociados al evento.</param>
    private async void OnRecordClicked(object sender, EventArgs e)
    {
        try
        {
            //var permission = await Permissions.RequestAsync<Permissions.Microphone>();

            //if (permission != PermissionStatus.Granted)
            //{
            //    await DisplayAlertAsync(
            //        "Permiso requerido",
            //        "Debes permitir el acceso al micrófono.",
            //        "Aceptar");
            //    return;
            //}

            await _audioRecordingService.StartRecordingAsync();

            _isRecording = true;
            RecordingStatus.Text = "Grabando audio...";
            RecordButton.IsEnabled = false;
            StopButton.IsEnabled = true;
            DiscardButton.IsEnabled = false;
            ConvertButton.IsEnabled = false;
            AudioFileStatus.Text = "Grabación en curso";
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Error", ex.Message, "Aceptar");
        }
    }
    /// <summary>
    /// Detiene la grabación de audio y actualiza el estado de la interfaz.
    /// </summary>
    /// <param name="sender">Control que generó el evento.</param>
    /// <param name="e">Datos del evento.</param>
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

    /// <summary>
    /// Descarta la grabación
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private async void OnDiscardClicked(object sender, EventArgs e)
    {
        try
        {
            await _audioRecordingService.DiscardAsync();

            RecordingStatus.Text = "Listo para grabar";
            AudioFileStatus.Text = "No hay audio grabado";

            RecordButton.IsEnabled = true;
            StopButton.IsEnabled = false;
            DiscardButton.IsEnabled = false;
            ConvertButton.IsEnabled = false;
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Error", ex.Message, "Aceptar");
        }
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


    /// <summary>
    /// Convierte la grabación en archivos MIDI y MusicXML y carga la partitura.
    /// </summary>
    /// <param name="sender">Control que generó el evento.</param>
    /// <param name="e">Datos del evento.</param>
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

        ConvertButton.IsEnabled = false;
        RecordButton.IsEnabled = false;
        StopButton.IsEnabled = false;
        DiscardButton.IsEnabled = false;

        try
        {
            // Lee los ajustes antes de iniciar las tareas de análisis y limpieza.
            var (meterNumeratorOverride, meterDenominatorOverride) =
                ReadMeterOverride();
            double? tempoOverride = ReadTempoOverride();
            string? tempoUnitOverride = ReadTempoUnitOverride();

            RecordingStatus.Text = "Transcribiendo con Magenta...";

            string outputDirectory = Path.Combine(
                FileSystem.AppDataDirectory,
                "MidiOutput",
                DateTime.Now.ToString("yyyyMMdd_HHmmss"));

            Directory.CreateDirectory(outputDirectory);

            string audioPath = _audioFilePath;

            string midiPath = await _magentaService.TranscribeAsync(
                audioPath,
                outputDirectory);

            _midiFilePath = midiPath;

            RecordingStatus.Text = "Analizando audio y MIDI...";

            var analysis = await _musicAnalysisService.AnalyzeAsync(
                audioPath,
                midiPath,
                outputDirectory);

            RecordingStatus.Text =
                "Evaluando las notas MIDI contra el audio...";

            MidiCleanupResult cleanupResult =
    await _midiCleanupService.CleanAsync(
        audioPath,
        midiPath,
        outputDirectory,
        analysis,
        meterNumeratorOverride,
        meterDenominatorOverride,
        tempoOverride,
        tempoUnitOverride);

            string cleanedMidiPath =
                cleanupResult.CleanedMidiPath
                ?? throw new InvalidOperationException(
                    "El limpiador no devolvió la ruta del MIDI limpio.");

            if (!File.Exists(cleanedMidiPath))
            {
                throw new FileNotFoundException(
                    "No se encontró el MIDI limpio.",
                    cleanedMidiPath);
            }

            // El botón Play reproducirá la versión limpia.
            _midiFilePath = cleanedMidiPath;

            string cleanupSummary =
                $"Limpieza MIDI: {cleanupResult.TotalNotes} notas analizadas; " +
                $"{cleanupResult.RemovedCount} eliminadas; " +
                $"{cleanupResult.SupportedCount} respaldadas por el audio; " +
                $"{cleanupResult.ReviewCount} para revisar; " +
                $"{cleanupResult.InconclusiveCount} inconclusas." +
                $"{Environment.NewLine}MIDI limpio: {cleanedMidiPath}" +
                $"{Environment.NewLine}Reporte: {cleanupResult.ReportPath}";

            string librosaTempo =
                analysis.TempoBpm?.ToString("F1") ?? "sin dato";

            string madmomTempo =
                analysis.MadmomTempoBpm?.ToString("F1") ?? "sin dato";

            string tempoCandidates =
                analysis.TempoCandidatesBpm is { Count: > 0 }
                    ? string.Join(
                        ", ",
                        analysis.TempoCandidatesBpm.Select(
                            value => $"{value:F1} BPM"))
                    : "sin candidatos";

            string TempoUnitLabel(string? unit) => unit switch
            {
                "eighth" => "corchea",
                "quarter" => "negra",
                "dotted_quarter" => "negra con puntillo",
                "half" => "blanca",
                _ => "unidad no identificada"
            };
            string selectedTempo =
                analysis.SelectedTempoBpm is double selectedBpm
                    ? $"{selectedBpm:F1} BPM " +
                      $"({TempoUnitLabel(analysis.SelectedTempoUnit)})"
                    : "ambiguo; revisar hipótesis";

            string tempoHypotheses =
                analysis.TempoHypotheses is { Count: > 0 }
                    ? string.Join(
                        Environment.NewLine,
                        analysis.TempoHypotheses
                            .Take(6)
                            .Select(hypothesis =>
                            {
                                string pulseDescription =
                                    hypothesis.GridPulseBpm is double pulseBpm
                                        ? $"; cuadrícula {pulseBpm:F1} BPM " +
                                          $"({TempoUnitLabel(hypothesis.GridPulseUnit)}), " +
                                          $"{hypothesis.GridPulsesPerBar} pulsos/compás"
                                        : "";

                                return
                                    $"  {hypothesis.TempoBpm:F1} BPM " +
                                    $"({TempoUnitLabel(hypothesis.TempoUnit)}), " +
                                    $"{hypothesis.Numerator}/{hypothesis.Denominator}, " +
                                    $"puntaje {hypothesis.TotalScore:F3}" +
                                    pulseDescription +
                                    (hypothesis.IsSelected ? " (seleccionada)" : "");
                            }))
                    : "  No hay hipótesis evaluables";

            string selectedMeter =
                analysis.TimeSignatureNumerator is int numerator &&
                analysis.TimeSignatureDenominator is int denominator
                    ? $"{numerator}/{denominator}"
                    : "sin confirmar";

            string meterCandidates =
                analysis.MeterCandidates is { Count: > 0 }
                    ? string.Join(
                        Environment.NewLine,
                        analysis.MeterCandidates
                            .OrderByDescending(candidate => candidate.Score)
                            .Select(candidate =>
                                $"  {candidate.Numerator}/{candidate.Denominator} " +
                                $"(puntaje {candidate.Score:F3})"))
                    : "  No hay candidatos";

            string meterMargin =
                analysis.MeterConfidence is double margin
                    ? margin.ToString("F3")
                    : "no disponible";

            string tempoEstimateSourceLabel =
                analysis.TempoEstimateSource switch
                {
                    "librosa_madmom_consensus" =>
                        "acuerdo entre Librosa y Madmom",
                    "librosa_only" =>
                        "solo Librosa disponible",
                    "madmom_only" =>
                        "solo Madmom disponible",
                    "estimators_disagree" =>
                        "Librosa y Madmom discrepan",
                    _ => "sin estimación"
                };

            string independentTempo =
                analysis.EstimatedTempoBpm is double estimatedBpm
                    ? $"{estimatedBpm:F1} BPM " +
                      $"({tempoEstimateSourceLabel})"
                    : tempoEstimateSourceLabel;

            if (analysis.TempoEstimatorAgreement is double agreement)
            {
                independentTempo +=
                    $"; cercanía {agreement:P1}";
            }

            string analysisSummary =
                $"Tempo Librosa: {librosaTempo} BPM" +
                $"{Environment.NewLine}Tempo Madmom: {madmomTempo} BPM" +
                $"{Environment.NewLine}Tempos candidatos: {tempoCandidates}" +
                $"{Environment.NewLine}Compás seleccionado: {selectedMeter}" +
                $"{Environment.NewLine}Margen de compás: {meterMargin}" +
                $"{Environment.NewLine}Decisión de compás: {analysis.MeterDecisionReason}" +
                $"{Environment.NewLine}Candidatos de compás:" +
                $"{Environment.NewLine}{meterCandidates}" +
                $"{Environment.NewLine}{Environment.NewLine}{cleanupSummary}" +
                $"{Environment.NewLine}Tempo seleccionado: {selectedTempo}" +
                $"{Environment.NewLine}Decisión de tempo: {analysis.TempoDecisionReason}" +
                $"{Environment.NewLine}Hipótesis tempo/compás:" +
                $"{Environment.NewLine}{tempoHypotheses}"+
                $"{Environment.NewLine}Tempo numérico independiente: {independentTempo}" +
                $"{Environment.NewLine}Compás usado en MusicXML: " +
                (meterNumeratorOverride is int manualNumerator &&
                 meterDenominatorOverride is int manualDenominator
                    ? $"{manualNumerator}/{manualDenominator} (manual)"
                    : $"{selectedMeter} (análisis) / predeterminado si no se confirmó") +
                $"{Environment.NewLine}Tempo usado en MusicXML: " +
                (tempoOverride is double manualTempo
                    ? $"{manualTempo:F1} BPM (manual, " +
                      $"{TempoUnitLabel(tempoUnitOverride ?? "quarter")})"
                    : $"{selectedTempo} (análisis)");

            RecordingStatus.Text =
                $"Análisis completado.{Environment.NewLine}{analysisSummary}";

            string musicXmlPath =
                await _musicXmlService.ConvertAsync(
                    cleanedMidiPath,
                    outputDirectory,
                    analysis,
                    meterNumeratorOverride,
                    meterDenominatorOverride,
                    tempoOverride,
                    tempoUnitOverride);

            string scoreViewerDirectory = Path.Combine(
                FileSystem.AppDataDirectory,
                "ScoreViewer");

            Directory.CreateDirectory(scoreViewerDirectory);

            string viewerMusicXmlPath = Path.Combine(
                scoreViewerDirectory,
                "grabacion.musicxml");

            File.Copy(musicXmlPath, viewerMusicXmlPath, overwrite: true);

            await LoadScoreViewerAsync("grabacion.musicxml");

            RecordingStatus.Text =
                $"Transcripción completada.{Environment.NewLine}{analysisSummary}";

            await DisplayAlertAsync(
                "Conversión completada",
                $"MIDI original:{Environment.NewLine}{midiPath}" +
                $"{Environment.NewLine}{Environment.NewLine}" +
                $"MIDI limpio:{Environment.NewLine}{cleanedMidiPath}" +
                $"{Environment.NewLine}{Environment.NewLine}" +
                $"MusicXML:{Environment.NewLine}{musicXmlPath}" +
                $"{Environment.NewLine}{Environment.NewLine}" +
                analysisSummary,
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


    /// <summary>
    /// Copia recursivamente un directorio y su contenido a otra ubicación.
    /// </summary>
    /// <param name="sourceDirectory">Ruta del directorio de origen.</param>
    /// <param name="destinationDirectory">Ruta del directorio de destino.</param>
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

    /// <summary>
    /// Hoja donde estara MIDI, prepara el visor
    /// </summary>
    /// <returns></returns>
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
            //if (File.Exists(destinationPath))
            //    continue;

            string assetPath = $"ScoreViewer/{file}";

            await using Stream source =
                await FileSystem.OpenAppPackageFileAsync(assetPath);

            await using FileStream destination = File.Create(destinationPath);

            await source.CopyToAsync(destination);
        }

        return Path.Combine(destinationDirectory, "index.html");
    }

    /// <summary>
    /// Carga el MIDI al visor
    /// </summary>
    /// <param name="fileName"></param>
    /// <returns></returns>
    /// <exception cref="FileNotFoundException"></exception>
    private async Task LoadScoreViewerAsync(
      string fileName = "prueba.musicxml")
    {
        try
        {
            ////DebugStatus.Text = "Preparando archivos del visor...";

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

            ////DebugStatus.Text =$"Archivos encontrados. MusicXML: {xmlSize} bytes. Cargando visor...";

#if WINDOWS
            var nativeWebView =
                (Microsoft.UI.Xaml.Controls.WebView2)ScoreWebView.Handler!.PlatformView;

            await nativeWebView.EnsureCoreWebView2Async();

            nativeWebView.CoreWebView2.SetVirtualHostNameToFolderMapping(
                "scoreviewer.example",
                directory,
                Microsoft.Web.WebView2.Core.CoreWebView2HostResourceAccessKind.DenyCors);

            ScoreWebView.Source = new UrlWebViewSource
            {
                Url =
                    $"https://scoreviewer.example/index.html" +
                    $"?file={Uri.EscapeDataString(fileName)}" +
                    $"&t={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"
            };
#endif

        }
        catch (Exception ex)
        {
            ////DebugStatus.Text = $"Error: {ex.Message}";
        }
    }


    /// <summary>
    /// Solo carga la partitura de ejemplo al abrir la página. Despues carga la grabacion.
    /// </summary>
    protected override void OnAppearing()
    {
        base.OnAppearing();

        ////DebugStatus.Text = "OnAppearing ejecutado";

        //if (ScoreWebView.Source is null)
        //{
        //    _ = LoadScoreViewerAsync("prueba.musicxml");
        //}
    }


    private void ScoreWebView_Navigated(
        object? sender,
        WebNavigatedEventArgs e)
    {
        ////DebugStatus.Text = $"navegación del visor: {e.Result}\n{e.Url}";
    }

    /// <summary>
    /// Boton de Play
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private async void OnPlayClicked(object sender, EventArgs e)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_midiFilePath))
            {
                await DisplayAlertAsync("MIDI", "Primero graba y convierte el audio.", "Aceptar");
                return;
            }

            PlayButton.IsEnabled = false;
            StopPlaybackButton.IsEnabled = true;

            await _midiPlaybackService.PlayAsync(_midiFilePath);
        }
        catch (Exception ex)
        {
            await DisplayAlertAsync("Error de reproducción", ex.Message, "Aceptar");
        }
        finally
        {
            PlayButton.IsEnabled =
                !string.IsNullOrWhiteSpace(_midiFilePath) &&
                File.Exists(_midiFilePath);

            StopPlaybackButton.IsEnabled = false;
        }
    }

    /// <summary>
    /// Boton de Stop
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void OnStopPlaybackClicked(object sender, EventArgs e)
    {
        _midiPlaybackService.Stop();
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

