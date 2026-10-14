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
            window.Title = "PRUEBA DE COMPILACIÓN - 12345";
            return window;
        }

    }
}