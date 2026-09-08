param(
    [int]$CheckIntervalSeconds = 5,
    [int]$FailuresBeforeRestart = 3,
    [int]$MaxHeartbeatAgeSeconds = 30,
    [int]$StartupGraceSeconds = 45,
    [int]$MaxRestartsPerWindow = 3,
    [int]$RestartWindowMinutes = 30,
    [int]$CircuitOpenSeconds = 300,
    [int]$ApiPort = 8080
)

$ErrorActionPreference = "Stop"
$workspace = [IO.Path]::GetFullPath(
    (Split-Path -Parent $MyInvocation.MyCommand.Path)
)
$entrypoint = [IO.Path]::GetFullPath((Join-Path $workspace "main.py"))
$logPath = Join-Path $workspace "watchdog.log"
$stdoutPath = Join-Path $workspace "runtime_stdout.log"
$stderrPath = Join-Path $workspace "runtime_stderr.log"
$apiUri = "http://127.0.0.1:$ApiPort/api/state"

function Resolve-PythonExecutable {
    $configured = [Environment]::GetEnvironmentVariable("LLM_TRADE_PYTHON")
    if (-not [string]::IsNullOrWhiteSpace($configured)) {
        $candidate = [IO.Path]::GetFullPath($configured)
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
        throw "LLM_TRADE_PYTHON does not point to an executable: $candidate"
    }

    $knownInstall = "C:\Python314\python.exe"
    if (Test-Path -LiteralPath $knownInstall -PathType Leaf) {
        return $knownInstall
    }

    $command = Get-Command python.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
        return [IO.Path]::GetFullPath($command.Source)
    }
    throw "Python executable was not found. Set LLM_TRADE_PYTHON to python.exe."
}

function Get-WorkspaceMutexName {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($workspace.ToLowerInvariant())
        $digest = $sha.ComputeHash($bytes)
        $shortHash = (
            [BitConverter]::ToString($digest, 0, 8) -replace "-", ""
        )
        return "Local\LLMTradeWatchdog-$shortHash"
    }
    finally {
        $sha.Dispose()
    }
}

$python = Resolve-PythonExecutable
$entrypointPattern = (
    "(?i)(?:^|[\s`"])" +
    [Regex]::Escape($entrypoint) +
    "(?:[\s`"]|$)"
)
$mutex = [Threading.Mutex]::new($false, (Get-WorkspaceMutexName))
$ownsMutex = $false

try {
    try {
        $ownsMutex = $mutex.WaitOne(0)
    }
    catch [Threading.AbandonedMutexException] {
        $ownsMutex = $true
    }
    if (-not $ownsMutex) {
        exit 0
    }

    function Write-WatchdogLog {
        param([string]$Message)
        $line = "{0} {1}" -f (
            Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        ), $Message
        Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    }

    function Write-StatusTransition {
        param(
            [string]$Signature,
            [string]$Message
        )
        if ($script:lastStatusSignature -ne $Signature) {
            Write-WatchdogLog $Message
            $script:lastStatusSignature = $Signature
        }
    }

    function Get-EngineProcesses {
        @(
            Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Name -match "^python(?:w)?(?:\d+(?:\.\d+)*)?\.exe$" -and
                    [string]$_.CommandLine -match $entrypointPattern
                }
        )
    }

    function Get-ListeningProcessId {
        try {
            $listener = Get-NetTCPConnection `
                -State Listen `
                -LocalPort $ApiPort `
                -ErrorAction Stop |
                Where-Object {
                    $_.LocalAddress -in @("127.0.0.1", "::1", "0.0.0.0", "::")
                } |
                Select-Object -First 1
            if ($listener) {
                return [int]$listener.OwningProcess
            }
        }
        catch {
            # Older Windows installations may not expose Get-NetTCPConnection.
        }
        return 0
    }

    function Test-ApiPortListening {
        $client = [Net.Sockets.TcpClient]::new()
        try {
            $pending = $client.BeginConnect("127.0.0.1", $ApiPort, $null, $null)
            if (-not $pending.AsyncWaitHandle.WaitOne(750)) {
                return $false
            }
            $client.EndConnect($pending)
            return $true
        }
        catch {
            return $false
        }
        finally {
            $client.Dispose()
        }
    }

    function Get-ServiceState {
        $state = Invoke-RestMethod -Uri $apiUri -TimeoutSec 4
        if (
            $null -eq $state.engine_running -or
            $null -eq $state.automation -or
            $null -eq $state.readiness
        ) {
            throw "The service returned an invalid /api/state payload"
        }
        return $state
    }

    function Get-ReadinessCheck {
        param(
            $State,
            [string]$Code
        )
        @($State.readiness.checks) |
            Where-Object { [string]$_.code -eq $Code } |
            Select-Object -First 1
    }

    function Get-HealthAssessment {
        param($State)

        $restartReasons = [Collections.Generic.List[string]]::new()
        $dependencyIssues = [Collections.Generic.List[string]]::new()

        if (-not [bool]$State.engine_running) {
            $restartReasons.Add("decision engine reports stopped")
        }

        $connectionCheck = Get-ReadinessCheck $State "CONNECTION"
        $brokerDisconnected = (
            $connectionCheck -and -not [bool]$connectionCheck.ok
        )

        $heartbeatText = [string]$State.automation.broker_poll_heartbeat_utc
        if ([string]::IsNullOrWhiteSpace($heartbeatText)) {
            $restartReasons.Add("broker-position heartbeat is missing")
        }
        else {
            try {
                $heartbeat = [DateTimeOffset]::Parse($heartbeatText)
                $heartbeatAge = (
                    [DateTimeOffset]::UtcNow - $heartbeat.ToUniversalTime()
                ).TotalSeconds
                $heartbeatLimit = [Math]::Max(
                    10, $MaxHeartbeatAgeSeconds
                )
                if ($heartbeatAge -lt -5 -or $heartbeatAge -gt $heartbeatLimit) {
                    $restartReasons.Add(
                        "broker-position heartbeat is stale " +
                        "($([Math]::Round($heartbeatAge, 1))s; " +
                        "limit ${heartbeatLimit}s)"
                    )
                }
            }
            catch {
                $restartReasons.Add("broker-position heartbeat is invalid")
            }
        }

        foreach ($code in @("LOOP", "PROTECTION")) {
            $check = Get-ReadinessCheck $State $code
            if ($check -and -not [bool]$check.ok) {
                $detail = "$($check.label): $($check.detail)"
                if ($brokerDisconnected -and $code -eq "PROTECTION") {
                    $dependencyIssues.Add($detail)
                }
                else {
                    $restartReasons.Add($detail)
                }
            }
        }

        foreach ($code in @(
            "ACCOUNT",
            "HISTORY",
            "CONNECTION",
            "PERMISSIONS",
            "POSITIONS",
            "LLM"
        )) {
            $check = Get-ReadinessCheck $State $code
            if ($check -and -not [bool]$check.ok) {
                $dependencyIssues.Add("$($check.label): $($check.detail)")
            }
        }

        if (-not [bool]$State.automation.decision_provider_inference_ready) {
            $dependencyIssues.Add(
                "decision provider is not inference-ready"
            )
        }

        return [pscustomobject]@{
            RestartRequired = $restartReasons.Count -gt 0
            RestartReason = ($restartReasons | Select-Object -Unique) -join "; "
            DependencyIssues = @(
                $dependencyIssues | Select-Object -Unique
            )
        }
    }

    function Remove-ExpiredRestartHistory {
        $cutoff = [DateTimeOffset]::UtcNow.AddMinutes(
            -[Math]::Max(1, $RestartWindowMinutes)
        )
        $script:restartHistory = @(
            $script:restartHistory | Where-Object { $_ -ge $cutoff }
        )
    }

    function Approve-RecoveryRestart {
        Remove-ExpiredRestartHistory
        $now = [DateTimeOffset]::UtcNow
        if ($now -lt $script:circuitOpenUntil) {
            return $false
        }
        if (
            $script:restartHistory.Count -ge
            [Math]::Max(1, $MaxRestartsPerWindow)
        ) {
            $openSeconds = [Math]::Max(30, $CircuitOpenSeconds)
            $windowMinutes = [Math]::Max(1, $RestartWindowMinutes)
            $script:circuitOpenUntil = $now.AddSeconds($openSeconds)
            $message = (
                "Restart circuit opened for {0}s after {1} recoveries " +
                "within {2} minutes. The service will not be recycled " +
                "during this interval."
            ) -f (
                $openSeconds,
                $script:restartHistory.Count,
                $windowMinutes
            )
            Write-StatusTransition "CIRCUIT_OPEN" $message
            return $false
        }
        $script:restartHistory += $now
        return $true
    }

    function Start-TradingService {
        $process = Start-Process `
            -FilePath $python `
            -ArgumentList "`"$entrypoint`"" `
            -WorkingDirectory $workspace `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        $script:startupDeadline = [DateTimeOffset]::UtcNow.AddSeconds(
            [Math]::Max(10, $StartupGraceSeconds)
        )
        $script:knownServiceObserved = $true
        $script:lastManagedProcessId = [int]$process.Id
        $script:consecutiveFailures = 0
        $script:lastStatusSignature = ""
        Write-WatchdogLog "Started trading service PID $($process.Id)."
    }

    function Get-RestartTarget {
        param($EngineProcesses)
        $processes = @($EngineProcesses)
        if ($processes.Count -eq 0) {
            return $null
        }
        $listenerProcessId = Get-ListeningProcessId
        if ($listenerProcessId -gt 0) {
            $owner = $processes |
                Where-Object { [int]$_.ProcessId -eq $listenerProcessId } |
                Select-Object -First 1
            if ($owner) {
                return $owner
            }
        }
        if ($processes.Count -eq 1) {
            return $processes[0]
        }
        return $null
    }

    if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
        throw "Trading entrypoint not found at $entrypoint"
    }

    $script:consecutiveFailures = 0
    $script:restartHistory = @()
    $script:circuitOpenUntil = [DateTimeOffset]::MinValue
    $script:startupDeadline = [DateTimeOffset]::MinValue
    $script:knownServiceObserved = $false
    $script:replacementAuthorized = $false
    $script:lastManagedProcessId = 0
    $script:lastStatusSignature = ""
    Write-WatchdogLog (
        "Watchdog started for $entrypoint (Python: $python)."
    )

    while ($true) {
        $engineProcesses = @(Get-EngineProcesses)
        if ($engineProcesses.Count -gt 0) {
            if (-not $script:knownServiceObserved) {
                try {
                    $createdAt = [DateTimeOffset]$engineProcesses[0].CreationDate
                    $adoptionDeadline = $createdAt.AddSeconds(
                        [Math]::Max(10, $StartupGraceSeconds)
                    )
                    if ($adoptionDeadline -gt [DateTimeOffset]::UtcNow) {
                        $script:startupDeadline = $adoptionDeadline
                    }
                }
                catch {
                    # A failed age lookup simply means no adoption grace.
                }
            }
            $script:knownServiceObserved = $true
        }
        if ($engineProcesses.Count -gt 1) {
            $duplicateMessage = (
                "Multiple matching engine processes were found ({0}). " +
                "No automatic process termination will be attempted."
            ) -f (($engineProcesses.ProcessId -join ", "))
            Write-StatusTransition "DUPLICATE_PROCESSES" $duplicateMessage
        }

        $state = $null
        $apiError = ""
        try {
            $state = Get-ServiceState
            $script:knownServiceObserved = $true
        }
        catch {
            $apiError = $_.Exception.Message
        }

        if ($null -eq $state) {
            if ($engineProcesses.Count -eq 0) {
                if (Test-ApiPortListening) {
                    Write-StatusTransition `
                        "PORT_CONFLICT" `
                        (
                            "Port $ApiPort is occupied but the trading API did " +
                            "not return valid state. A second engine will not " +
                            "be started."
                        )
                }
                else {
                    if ($script:replacementAuthorized) {
                        $startApproved = $true
                        $script:replacementAuthorized = $false
                    }
                    else {
                        $startApproved = -not $script:knownServiceObserved
                    }
                    if (
                        -not $startApproved -and
                        -not $script:replacementAuthorized
                    ) {
                        $startApproved = Approve-RecoveryRestart
                    }
                    if ($startApproved) {
                        Start-TradingService
                    }
                    else {
                        Write-StatusTransition `
                            "RECOVERY_SUPPRESSED" `
                            (
                                "Trading service is down; automatic launch is " +
                                "temporarily suppressed by the restart circuit."
                            )
                    }
                }
            }
            elseif (
                [DateTimeOffset]::UtcNow -lt $script:startupDeadline
            ) {
                Write-StatusTransition `
                    "STARTING" `
                    "Trading service is starting; API grace period is active."
            }
            else {
                $script:consecutiveFailures += 1
                $failureThreshold = [Math]::Max(
                    1, $FailuresBeforeRestart
                )
                if ($script:consecutiveFailures -le $failureThreshold) {
                    Write-WatchdogLog (
                        "API health check failed ({0}/{1}): {2}" -f
                        $script:consecutiveFailures,
                        $failureThreshold,
                        $apiError
                    )
                }
                if (
                    $script:consecutiveFailures -ge
                    $failureThreshold
                ) {
                    $target = Get-RestartTarget $engineProcesses
                    if ($target -and (Approve-RecoveryRestart)) {
                        Stop-Process -Id $target.ProcessId -Force
                        $script:replacementAuthorized = $true
                        Write-WatchdogLog (
                            "Stopped API-unresponsive trading service PID " +
                            "$($target.ProcessId); a clean replacement will start."
                        )
                        $script:consecutiveFailures = 0
                    }
                    elseif (-not $target) {
                        Write-StatusTransition `
                            "UNMANAGED_API_FAILURE" `
                            (
                                "The unresponsive API process could not be " +
                                "identified safely; no process was terminated."
                            )
                    }
                    else {
                        Write-StatusTransition `
                            "API_RECOVERY_SUPPRESSED" `
                            "API recovery is temporarily suppressed by the restart circuit."
                    }
                }
            }

            Start-Sleep -Seconds ([Math]::Max(2, $CheckIntervalSeconds))
            continue
        }

        $assessment = Get-HealthAssessment $state
        if (-not $assessment.RestartRequired) {
            $script:consecutiveFailures = 0
            if ($assessment.DependencyIssues.Count -gt 0) {
                Write-StatusTransition `
                    ("DEPENDENCY:" + ($assessment.DependencyIssues -join "|")) `
                    (
                        "Service loops are healthy; external/configuration " +
                        "dependency is unavailable (no process restart): " +
                        ($assessment.DependencyIssues -join "; ")
                    )
            }
            else {
                Write-StatusTransition `
                    "HEALTHY" `
                    "Trading API, decision loop, broker polling, and protection supervisor are healthy."
            }
        }
        elseif ([DateTimeOffset]::UtcNow -lt $script:startupDeadline) {
            Write-StatusTransition `
                "WARMING" `
                (
                    "Trading service is within its startup grace period: " +
                    $assessment.RestartReason
                )
        }
        else {
            $script:consecutiveFailures += 1
            $failureThreshold = [Math]::Max(1, $FailuresBeforeRestart)
            if ($script:consecutiveFailures -le $failureThreshold) {
                Write-WatchdogLog (
                    "Internal health check failed ({0}/{1}): {2}" -f
                    $script:consecutiveFailures,
                    $failureThreshold,
                    $assessment.RestartReason
                )
            }
            if (
                $script:consecutiveFailures -ge
                $failureThreshold
            ) {
                $target = Get-RestartTarget $engineProcesses
                if ($target -and (Approve-RecoveryRestart)) {
                    Stop-Process -Id $target.ProcessId -Force
                    $script:replacementAuthorized = $true
                    Write-WatchdogLog (
                        "Restarting stalled trading service PID " +
                        "$($target.ProcessId): $($assessment.RestartReason)"
                    )
                    $script:consecutiveFailures = 0
                }
                elseif (-not $target) {
                    Write-StatusTransition `
                        "UNMANAGED_INTERNAL_FAILURE" `
                        (
                            "The unhealthy API is reachable, but its process " +
                            "does not match this workspace. No process was " +
                            "terminated and no duplicate engine was started."
                        )
                }
                else {
                    Write-StatusTransition `
                        "INTERNAL_RECOVERY_SUPPRESSED" `
                        "Internal recovery is temporarily suppressed by the restart circuit."
                }
            }
        }

        Start-Sleep -Seconds ([Math]::Max(2, $CheckIntervalSeconds))
    }
}
catch {
    try {
        $line = "{0} Watchdog stopped: {1}" -f (
            Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        ), $_.Exception.Message
        Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    }
    catch {
        # Preserve the original exception if the log itself is unavailable.
    }
    throw
}
finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
