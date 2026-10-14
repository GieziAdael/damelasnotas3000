
#if WINDOWS
using Microsoft.Maui.Handlers;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Web.WebView2.Core;

namespace Damelasnotas3000.Platforms.Windows;

public class ScoreViewerWebViewHandler : WebViewHandler
{
    protected override WebView2 CreatePlatformView()
    {
        var webView = base.CreatePlatformView();

        webView.CoreWebView2Initialized += (_, args) =>
        {
            if (args.Exception is not null)
                return;

            string scoreDirectory = Path.Combine(
                FileSystem.AppDataDirectory,
                "ScoreViewer");

            Directory.CreateDirectory(scoreDirectory);

            webView.CoreWebView2.SetVirtualHostNameToFolderMapping(
                "damelasnotas3000.example",
                scoreDirectory,
                CoreWebView2HostResourceAccessKind.DenyCors);
        };

        return webView;
    }
}
#endif
