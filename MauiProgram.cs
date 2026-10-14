using Microsoft.Extensions.Logging;

namespace Damelasnotas3000
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
    		builder.Logging.AddDebug();
#endif

    #if WINDOWS
    builder.ConfigureMauiHandlers(handlers =>
    {
        handlers.AddHandler<
            Microsoft.Maui.Controls.WebView,
            Damelasnotas3000.Platforms.Windows.ScoreViewerWebViewHandler>();
    });
    #endif

            return builder.Build();
        }
    }
}
