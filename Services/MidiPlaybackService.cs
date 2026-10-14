using NAudio.Midi;

namespace Damelasnotas3000.Services;

public class MidiPlaybackService
{
    private CancellationTokenSource? _playbackCts;
    private MidiOut? _midiOut;
    private bool _isPlaying;

    public async Task PlayAsync(string midiFilePath)
    {
#if WINDOWS
        if (string.IsNullOrWhiteSpace(midiFilePath) ||
            !File.Exists(midiFilePath))
        {
            throw new FileNotFoundException(
                "No se encontró el MIDI. Primero graba y convierte el audio.",
                midiFilePath);
        }

        if (_isPlaying)
            return;

        _isPlaying = true;
        _playbackCts = new CancellationTokenSource();

        try
        {
            CancellationToken token = _playbackCts.Token;
            var midiFile = new MidiFile(midiFilePath, false);
            _midiOut = new MidiOut(0);

            var events = midiFile.Events
                .SelectMany(track => track)
                .Where(e => e is NoteOnEvent ||
                            e is NoteEvent ||
                            e is TempoEvent)
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

                if (midiEvent is TempoEvent tempoEvent)
                    tempo = tempoEvent.MicrosecondsPerQuarterNote;
                else if (midiEvent is NoteOnEvent noteOn)
                    _midiOut.Send(noteOn.GetAsShortMessage());
                else if (midiEvent is NoteEvent noteEvent)
                    _midiOut.Send(noteEvent.GetAsShortMessage());
            }
        }
        catch (OperationCanceledException)
        {
            // El usuario detuvo la reproducción.
        }
        finally
        {
            _midiOut?.Dispose();
            _midiOut = null;

            _playbackCts?.Dispose();
            _playbackCts = null;

            _isPlaying = false;
        }
#else
        throw new PlatformNotSupportedException(
            "La reproducción MIDI está disponible solo en Windows.");
#endif
    }

    public void Stop()
    {
        _playbackCts?.Cancel();
    }
}