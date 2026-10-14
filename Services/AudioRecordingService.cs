using Plugin.Maui.Audio;

namespace Damelasnotas3000.Services
{
    public class AudioRecordingService
    {
        private readonly IAudioRecorder _recorder;
        private bool _isRecording;
        private string? _audioFilePath;

        public AudioRecordingService(IAudioRecorder recorder)
        {
            _recorder = recorder;
        }

        public Task StartRecordingAsync()
        {
            return _recorder.StartAsync();
        }

        public async Task DiscardAsync()
        {
            if (_isRecording)
            {
                await _recorder.StopAsync();
                _isRecording = false;
            }

            if (!string.IsNullOrWhiteSpace(_audioFilePath)
                && File.Exists(_audioFilePath))
            {
                File.Delete(_audioFilePath);
            }

            _audioFilePath = null;
        }
    }
}
