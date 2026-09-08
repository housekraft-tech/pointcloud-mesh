param([Parameter(Mandatory=$true)][string]$Destination)

Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class SketchUpWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out RECT rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
'@
[SketchUpWindowCapture]::SetProcessDPIAware() | Out-Null
$sketchProcess = Get-Process SketchUp -ErrorAction Stop | Where-Object MainWindowHandle -NE 0 | Select-Object -First 1
$windowRect = New-Object SketchUpWindowCapture+RECT
if (-not [SketchUpWindowCapture]::GetWindowRect($sketchProcess.MainWindowHandle, [ref]$windowRect)) { throw 'Cannot read SketchUp window bounds' }
$bitmap = New-Object System.Drawing.Bitmap(($windowRect.Right-$windowRect.Left), ($windowRect.Bottom-$windowRect.Top))
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$dc = $graphics.GetHdc()
try { $captured = [SketchUpWindowCapture]::PrintWindow($sketchProcess.MainWindowHandle, $dc, 2) }
finally { $graphics.ReleaseHdc($dc) }
try {
    if (-not $captured) { throw 'SketchUp window capture failed' }
    $bitmap.Save($Destination, [System.Drawing.Imaging.ImageFormat]::Png)
    [PSCustomObject]@{Image=$Destination; Title=$sketchProcess.MainWindowTitle; Width=$bitmap.Width; Height=$bitmap.Height} | ConvertTo-Json
}
finally { $graphics.Dispose(); $bitmap.Dispose() }
