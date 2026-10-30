using System.Diagnostics;
using System.Text;

namespace Damelasnotas3000.Services;

/// <summary>
/// Guarda en un TXT los datos y las salidas de cada etapa de una conversión.
/// Los fallos de escritura del diagnóstico no interrumpen la transcripción.
/// </summary>
public static class AnalysisDebugLog
{
    public const string FileName = "analisis_grabacion_debug.txt";

    private static readonly SemaphoreSlim WriteLock = new(1, 1);

    public static string GetPath(string outputDirectory) =>
        Path.Combine(outputDirectory, FileName);

    public static async Task StartAsync(
        string outputDirectory,
        string audioFilePath,
        string transcriptionEngine,
        bool compareTranscribers)
    {
        string audioDetails = File.Exists(audioFilePath)
            ? $"Archivo: {Path.GetFileName(audioFilePath)}{Environment.NewLine}" +
              $"Ruta: {audioFilePath}{Environment.NewLine}" +
              $"Tamaño: {new FileInfo(audioFilePath).Length} bytes{Environment.NewLine}" +
              $"Última modificación: {File.GetLastWriteTime(audioFilePath):O}"
            : $"Archivo: {audioFilePath} (no encontrado al iniciar)";

        string header =
            "ANÁLISIS DE GRABACIÓN - REGISTRO DE DEPURACIÓN" +
            Environment.NewLine +
            $"Inicio: {DateTimeOffset.Now:O}" +
            Environment.NewLine +
            $"Motor principal: {transcriptionEngine}" +
            Environment.NewLine +
            $"Comparar transcriptores: {(compareTranscribers ? "sí" : "no")}" +
            Environment.NewLine +
            $"Carpeta de resultados: {outputDirectory}" +
            Environment.NewLine +
            Environment.NewLine +
            "AUDIO DE ENTRADA" + Environment.NewLine +
            audioDetails + Environment.NewLine;

        try
        {
            Directory.CreateDirectory(outputDirectory);
            await WriteLock.WaitAsync();
            try
            {
                await File.WriteAllTextAsync(
                    GetPath(outputDirectory),
                    header,
                    new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            }
            finally
            {
                WriteLock.Release();
            }
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"No se pudo iniciar el TXT de análisis: {exception}");
        }
    }

    public static async Task AppendSectionAsync(
        string outputDirectory,
        string title,
        string? content)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            return;
        }

        string section =
            Environment.NewLine +
            "============================================================" +
            Environment.NewLine +
            $"[{DateTimeOffset.Now:O}] {title}" +
            Environment.NewLine +
            "============================================================" +
            Environment.NewLine +
            (string.IsNullOrWhiteSpace(content) ? "(sin salida)" : content.TrimEnd()) +
            Environment.NewLine;

        try
        {
            Directory.CreateDirectory(outputDirectory);
            await WriteLock.WaitAsync();
            try
            {
                await File.AppendAllTextAsync(
                    GetPath(outputDirectory),
                    section,
                    new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            }
            finally
            {
                WriteLock.Release();
            }
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"No se pudo escribir el TXT de análisis: {exception}");
        }
    }

    public static async Task AppendProcessOutputAsync(
        string outputDirectory,
        string title,
        int exitCode,
        string standardOutput,
        string standardError)
    {
        string content =
            $"Código de salida: {exitCode}{Environment.NewLine}" +
            $"--- SALIDA ESTÁNDAR ---{Environment.NewLine}" +
            (string.IsNullOrWhiteSpace(standardOutput)
                ? "(vacía)"
                : standardOutput.TrimEnd()) +
            Environment.NewLine +
            $"--- ERRORES / AVISOS ---{Environment.NewLine}" +
            (string.IsNullOrWhiteSpace(standardError)
                ? "(vacíos)"
                : standardError.TrimEnd());

        await AppendSectionAsync(outputDirectory, title, content);
    }

    public static async Task AppendFileAsync(
        string outputDirectory,
        string title,
        string filePath)
    {
        try
        {
            string content = File.Exists(filePath)
                ? await File.ReadAllTextAsync(filePath)
                : $"No se generó el archivo: {filePath}";
            await AppendSectionAsync(outputDirectory, title, content);
        }
        catch (Exception exception)
        {
            await AppendSectionAsync(
                outputDirectory,
                title,
                $"No se pudo leer {filePath}:{Environment.NewLine}{exception}");
        }
    }
}
