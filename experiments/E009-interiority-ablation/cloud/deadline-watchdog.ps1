param([Parameter(Mandatory=$true)][string]$PodId,
      [Parameter(Mandatory=$true)][string]$DeadlineUtc,
      [Parameter(Mandatory=$true)][string]$LogPath,
      [string]$Cli='D:/Interiority-V1/tools/runpodctl-2.14.0/runpodctl.exe')
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class E009Awake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
[E009Awake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
    $deadline=[DateTimeOffset]::Parse($DeadlineUtc)
    "Watchdog started for $PodId; deadline $deadline" | Out-File -LiteralPath $LogPath -Encoding utf8
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath ($LogPath+'.cancel')) { 'Cancelled after verified teardown' | Add-Content -LiteralPath $LogPath; exit 0 }
        Start-Sleep -Seconds 15
    }
    for ($attempt=1; $attempt -le 40; $attempt++) {
        $result=& $Cli pod delete $PodId 2>&1
        $result | Add-Content -LiteralPath $LogPath
        if ($LASTEXITCODE -eq 0) { exit 0 }
        Start-Sleep -Seconds 15
    }
    throw 'Deadline termination retries exhausted'
} finally { [E009Awake]::SetThreadExecutionState([uint32]2147483648) | Out-Null }
