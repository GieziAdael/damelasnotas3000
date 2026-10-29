using System.Diagnostics;
using NAudio.Midi;

namespace Damelasnotas3000.Services;

public sealed class MidiPlaybackService
{
    private readonly object _gate = new();
    private string? _loadedMidiPath;
    private List<ScheduledMidiEvent> _scheduledEvents = [];
    private List<TempoSegment> _tempoSegments = [];
    private MidiOut? _midiOut;
    private CancellationTokenSource? _playbackCts;
    private Task? _playbackTask;
    private bool _isPlaying;
    private int _ticksPerQuarterNote = 480;
    private long _lastMidiTick;
    private double _positionSeconds;
    private double _durationSeconds;
    private double _segmentStartSeconds;
    private long _segmentStartTimestamp;

    public event Action<Exception>? PlaybackFailed;

    public bool HasLoadedMidi
    {
        get
        {
            lock (_gate)
                return _loadedMidiPath is not null;
        }
    }

    public string? LoadedMidiPath
    {
        get
        {
            lock (_gate)
                return _loadedMidiPath;
        }
    }

    public bool IsPlaying
    {
        get
        {
            lock (_gate)
                return _isPlaying;
        }
    }

    public int TicksPerQuarterNote
    {
        get
        {
            lock (_gate)
                return _ticksPerQuarterNote;
        }
    }

    public double DurationSeconds
    {
        get
        {
            lock (_gate)
                return _durationSeconds;
        }
    }

    public double PositionSeconds
    {
        get
        {
            lock (_gate)
                return GetPositionSecondsLocked();
        }
    }

    public long PositionTicks
    {
        get
        {
            lock (_gate)
                return SecondsToTicksLocked(GetPositionSecondsLocked());
        }
    }

    public async Task LoadAsync(string midiFilePath)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(midiFilePath) ||
            !File.Exists(midiFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el MIDI para reproducir.",
                midiFilePath);
        }

        await PauseAsync();

        lock (_gate)
        {
            if (string.Equals(
                    _loadedMidiPath,
                    midiFilePath,
                    StringComparison.OrdinalIgnoreCase))
            {
                _positionSeconds = 0;
                return;
            }

            SendAllNotesOffLocked();
            _midiOut?.Dispose();
            _midiOut = null;

            var midiFile = new MidiFile(midiFilePath, false);
            int ticksPerQuarterNote = midiFile.DeltaTicksPerQuarterNote;
            if (ticksPerQuarterNote <= 0)
            {
                throw new InvalidDataException(
                    "El MIDI no contiene una resolución de pulso válida.");
            }

            var allEvents = midiFile.Events
                .SelectMany(track => track)
                .OrderBy(midiEvent => midiEvent.AbsoluteTime)
                .ThenBy(midiEvent => midiEvent is TempoEvent ? 0 : 1)
                .ToList();

            var tempoSegments = new List<TempoSegment>
            {
                new(0, 0, 500_000)
            };
            var scheduledEvents = new List<ScheduledMidiEvent>();
            long previousTick = 0;
            double elapsedSeconds = 0;
            int tempoMicrosecondsPerQuarter = 500_000;

            foreach (MidiEvent midiEvent in allEvents)
            {
                long absoluteTick = midiEvent.AbsoluteTime;
                long deltaTicks = Math.Max(0, absoluteTick - previousTick);
                elapsedSeconds += deltaTicks *
                    tempoMicrosecondsPerQuarter /
                    (double)ticksPerQuarterNote / 1_000_000.0;
                previousTick = absoluteTick;

                if (midiEvent is TempoEvent tempoEvent)
                {
                    tempoMicrosecondsPerQuarter = Math.Max(
                        1,
                        tempoEvent.MicrosecondsPerQuarterNote);

                    if (tempoSegments[^1].StartTick == absoluteTick)
                    {
                        tempoSegments[^1] = new TempoSegment(
                            absoluteTick,
                            elapsedSeconds,
                            tempoMicrosecondsPerQuarter);
                    }
                    else
                    {
                        tempoSegments.Add(new TempoSegment(
                            absoluteTick,
                            elapsedSeconds,
                            tempoMicrosecondsPerQuarter));
                    }

                    continue;
                }

                if (midiEvent is NoteOnEvent || midiEvent is NoteEvent)
                {
                    scheduledEvents.Add(new ScheduledMidiEvent(
                        midiEvent,
                        absoluteTick,
                        elapsedSeconds));
                }
            }

            long lastMidiTick = allEvents.Count == 0
                ? 0
                : Math.Max(0, allEvents[^1].AbsoluteTime);

            _loadedMidiPath = midiFilePath;
            _ticksPerQuarterNote = ticksPerQuarterNote;
            _lastMidiTick = lastMidiTick;
            _tempoSegments = tempoSegments;
            _scheduledEvents = scheduledEvents;
            _durationSeconds = TicksToSecondsLocked(lastMidiTick);
            _positionSeconds = 0;
        }
#else
        await Task.CompletedTask;
        throw new PlatformNotSupportedException(
            "La reproducción MIDI está disponible solo en Windows.");
#endif
    }

    public Task PlayAsync()
    {
#if WINDOWS
        lock (_gate)
        {
            if (_loadedMidiPath is null)
            {
                throw new InvalidOperationException(
                    "Primero convierte el audio para cargar un MIDI.");
            }

            if (_isPlaying)
                return Task.CompletedTask;

            if (_positionSeconds >= _durationSeconds - 0.001)
                _positionSeconds = 0;

            _midiOut ??= new MidiOut(0);
            _segmentStartSeconds = _positionSeconds;
            _segmentStartTimestamp = Stopwatch.GetTimestamp();
            _playbackCts = new CancellationTokenSource();
            CancellationToken token = _playbackCts.Token;
            _isPlaying = true;
            _playbackTask = Task.Run(
                () => PlaybackLoopAsync(
                    _segmentStartSeconds,
                    _segmentStartTimestamp,
                    token));
        }

        return Task.CompletedTask;
#else
        throw new PlatformNotSupportedException(
            "La reproducción MIDI está disponible solo en Windows.");
#endif
    }

    public async Task PauseAsync()
    {
#if WINDOWS
        Task? playbackTask;

        lock (_gate)
        {
            if (!_isPlaying)
                return;

            _positionSeconds = GetPositionSecondsLocked();
            _isPlaying = false;
            _playbackCts?.Cancel();
            playbackTask = _playbackTask;
        }

        if (playbackTask is not null)
        {
            try
            {
                await playbackTask;
            }
            catch (OperationCanceledException)
            {
                // La pausa cancela la espera del siguiente evento MIDI.
            }
        }

        SendAllNotesOff();
        CloseMidiOut();
#else
        await Task.CompletedTask;
#endif
    }

    public async Task SeekByAsync(double seconds)
    {
        if (!double.IsFinite(seconds))
            return;

        bool resumeAfterSeek = IsPlaying;
        if (resumeAfterSeek)
            await PauseAsync();

        lock (_gate)
        {
            _positionSeconds = Math.Clamp(
                _positionSeconds + seconds,
                0,
                _durationSeconds);
        }

        SendAllNotesOff();

        if (resumeAfterSeek && PositionSeconds < DurationSeconds - 0.001)
            await PlayAsync();
    }

    public async Task RestartAsync()
    {
        await PauseAsync();

        lock (_gate)
            _positionSeconds = 0;

        SendAllNotesOff();
        await PlayAsync();
    }

    public async Task GoToEndAsync()
    {
        await PauseAsync();

        lock (_gate)
            _positionSeconds = _durationSeconds;

        SendAllNotesOff();
        CloseMidiOut();
    }

    private async Task PlaybackLoopAsync(
        double startPositionSeconds,
        long startedAtTimestamp,
        CancellationToken cancellationToken)
    {
        try
        {
            int eventIndex = FindFirstEventAtOrAfter(startPositionSeconds);
            while (eventIndex < _scheduledEvents.Count)
            {
                ScheduledMidiEvent scheduledEvent = _scheduledEvents[eventIndex];
                await WaitUntilAsync(
                    scheduledEvent.Seconds - startPositionSeconds,
                    startedAtTimestamp,
                    cancellationToken);

                cancellationToken.ThrowIfCancellationRequested();
                if (scheduledEvent.Event is NoteOnEvent noteOn)
                {
                    _midiOut?.Send(noteOn.GetAsShortMessage());
                }
                else if (scheduledEvent.Event is NoteEvent noteEvent)
                {
                    _midiOut?.Send(noteEvent.GetAsShortMessage());
                }
                eventIndex++;
            }

            double remainingSeconds = DurationSeconds - startPositionSeconds;
            await WaitUntilAsync(
                remainingSeconds,
                startedAtTimestamp,
                cancellationToken);

            lock (_gate)
            {
                if (cancellationToken.IsCancellationRequested)
                    return;

                _positionSeconds = _durationSeconds;
                _isPlaying = false;
                _playbackTask = null;
                _playbackCts?.Dispose();
                _playbackCts = null;
            }

            SendAllNotesOff();
            CloseMidiOut();
        }
        catch (OperationCanceledException)
        {
            // Pausa o búsqueda: el transporte conserva la posición calculada.
        }
        catch (Exception exception)
        {
            lock (_gate)
            {
                _positionSeconds = GetPositionSecondsLocked();
                _isPlaying = false;
                _playbackTask = null;
                _playbackCts?.Dispose();
                _playbackCts = null;
            }

            SendAllNotesOff();
            CloseMidiOut();
            PlaybackFailed?.Invoke(exception);
        }
    }

    private static async Task WaitUntilAsync(
        double targetSeconds,
        long startedAtTimestamp,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double elapsedSeconds =
                Stopwatch.GetElapsedTime(startedAtTimestamp).TotalSeconds;
            double remainingMilliseconds =
                (targetSeconds - elapsedSeconds) * 1000.0;

            if (remainingMilliseconds <= 0)
                return;

            await Task.Delay(
                TimeSpan.FromMilliseconds(
                    Math.Max(1, Math.Min(remainingMilliseconds, 50))),
                cancellationToken);
        }
    }

    private int FindFirstEventAtOrAfter(double seconds)
    {
        int low = 0;
        int high = _scheduledEvents.Count;

        while (low < high)
        {
            int middle = low + (high - low) / 2;
            if (_scheduledEvents[middle].Seconds < seconds - 1e-6)
                low = middle + 1;
            else
                high = middle;
        }

        return low;
    }

    private double GetPositionSecondsLocked()
    {
        if (!_isPlaying)
            return Math.Clamp(_positionSeconds, 0, _durationSeconds);

        double elapsed = Stopwatch.GetElapsedTime(
            _segmentStartTimestamp).TotalSeconds;
        return Math.Clamp(
            _segmentStartSeconds + elapsed,
            0,
            _durationSeconds);
    }

    private long SecondsToTicksLocked(double seconds)
    {
        if (_tempoSegments.Count == 0)
            return 0;

        TempoSegment segment = _tempoSegments[0];
        foreach (TempoSegment candidate in _tempoSegments)
        {
            if (candidate.StartSeconds > seconds)
                break;
            segment = candidate;
        }

        double ticks = segment.StartTick +
            (seconds - segment.StartSeconds) * 1_000_000.0 *
            _ticksPerQuarterNote / segment.MicrosecondsPerQuarterNote;
        return Math.Clamp(
            (long)Math.Round(ticks),
            0,
            _lastMidiTick);
    }

    private double TicksToSecondsLocked(long ticks)
    {
        if (_tempoSegments.Count == 0)
            return 0;

        TempoSegment segment = _tempoSegments[0];
        foreach (TempoSegment candidate in _tempoSegments)
        {
            if (candidate.StartTick > ticks)
                break;
            segment = candidate;
        }

        return segment.StartSeconds +
            Math.Max(0, ticks - segment.StartTick) *
            segment.MicrosecondsPerQuarterNote /
            (double)_ticksPerQuarterNote / 1_000_000.0;
    }

    private void SendAllNotesOff()
    {
        lock (_gate)
            SendAllNotesOffLocked();
    }

    private void CloseMidiOut()
    {
        lock (_gate)
        {
            try
            {
                _midiOut?.Dispose();
            }
            catch (Exception exception)
            {
                Debug.WriteLine("No se pudo cerrar la salida MIDI: " + exception.Message);
            }

            _midiOut = null;
        }
    }

    private void SendAllNotesOffLocked()
    {
        if (_midiOut is null)
            return;

        for (int channel = 0; channel < 16; channel++)
        {
            int allNotesOffMessage = 0xB0 | channel | (123 << 8);
            try
            {
                _midiOut.Send(allNotesOffMessage);
            }
            catch (Exception exception)
            {
                Debug.WriteLine("No se pudo detener una nota MIDI: " + exception.Message);
                break;
            }
        }
    }

    private sealed record ScheduledMidiEvent(
        MidiEvent Event,
        long AbsoluteTick,
        double Seconds);

    private sealed record TempoSegment(
        long StartTick,
        double StartSeconds,
        int MicrosecondsPerQuarterNote);
}
