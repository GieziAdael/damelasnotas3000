using Microsoft.Extensions.DependencyInjection;

namespace Damelasnotas3000
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
        }


        protected override Window CreateWindow(IActivationState? activationState)
        {
            var window = new Window(new AppShell());
            window.Title = "Damelasnotas3000";
            return window;
        }

    }
}