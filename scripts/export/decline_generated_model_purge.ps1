# Decline only the optional purge prompt during saves of our generated models.
# This preserves unused user/source components and does not change preferences.
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$suProc = Get-Process SketchUp | Where-Object MainWindowHandle -ne 0 | Select-Object -First 1
if ($null -eq $suProc -or $suProc.MainWindowTitle -notmatch '^Soulace_(inner_outer_wall_planes_joined_floors|paired_wall_planes_floor_junctions_checked|complete_wall_planes_and_floor_junctions|full_house_wall_and_floor_planes)') {
    Write-Output 'No matching generated model window'
    exit
}
$suRoot = [System.Windows.Automation.AutomationElement]::FromHandle($suProc.MainWindowHandle)
$purgeLabel = $suRoot.FindFirst([System.Windows.Automation.TreeScope]::Descendants, (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, 'Purge Unused?')))
if ($null -eq $purgeLabel) {
    Write-Output 'No optional purge prompt'
    exit
}
$noButton = $suRoot.FindFirst([System.Windows.Automation.TreeScope]::Descendants, (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, 'No')))
if ($null -eq $noButton) { throw 'Expected No button missing' }
$noButton.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Write-Output 'Declined optional purge; all source resources retained'
