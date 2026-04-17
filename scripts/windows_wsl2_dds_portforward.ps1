<#
.SYNOPSIS
    Forward DDS/CycloneDDS UDP ports from the Windows host NIC into a WSL2 instance.
    Tailored for CLIENT_ID=3 (SPDP port 7418, data port 7419) but covers the full
    DDS port range so discovery and model-weight transfer both work.

.DESCRIPTION
    ── Why this script is needed ──────────────────────────────────────────────────
    Running "ifconfig" inside WSL2 shows the internal virtual IP (e.g. 172.29.13.112).
    Running "ipconfig" on Windows shows the real LAN IP  (e.g. 192.168.0.100).

    Docker --network host inside WSL2 binds to the WSL2 virtual IP, NOT the Windows
    LAN IP.  This breaks CycloneDDS discovery in two ways:

      Problem 1 — wrong advertised locator
        CycloneDDS advertises 172.29.13.112 as its RTPS address.
        The FL server (192.168.0.102) tries to send to 172.29.13.112 which is
        unreachable from the LAN.
        FIX → set  DDS_EXTERNAL_NETWORK_ADDRESS=192.168.0.100  in the container so
        CycloneDDS advertises the reachable Windows LAN IP.
        The distributed_client_gui.py "WSL2 mode" checkbox does this automatically.

      Problem 2 — inbound UDP never reaches WSL2
        When the server sends SPDP/RTPS to 192.168.0.100:7418 (CLIENT_ID=3 SPDP port)
        the packet arrives at the Windows NIC but Windows does NOT automatically
        forward UDP to WSL2 at 172.29.13.112.
        FIX → this script creates PowerShell background jobs that bind on the Windows
        LAN IP and forward every received datagram into the WSL2 instance.

    ── Port assignment for CLIENT_ID=3 ───────────────────────────────────────────
    CycloneDDS port formula (Domain 0, ParticipantIndex = ClientID + 1):
      CLIENT_ID=3  →  ParticipantIndex=4
      SPDP (meta unicast) : 7410 + 2 * 4 = 7418   ← critical for discovery
      Data (user unicast) : 7411 + 2 * 4 = 7419   ← needed for model weights
    The script relays 7400–7500 so both ports and any ephemeral RTPS ports are covered.

    ── Network layout (distributed_3pc.env) ──────────────────────────────────────
      192.168.0.102  Server + Client1  (Ubuntu)
      192.168.0.101  Client2           (Linux PC)
      192.168.0.100  Client3           (Windows/WSL2)  ← this machine

    ── Recommended permanent fix (Windows 11 22H2+) ─────────────────────────────
    Add to  %USERPROFILE%\.wslconfig  then run  wsl --shutdown :
        [wsl2]
        networkingMode=mirrored
    WSL2 will share the Windows network stack and see 192.168.0.100 directly.
    No relay script needed after that.

.PARAMETER WindowsLanIp
    Windows host LAN IP (default: auto-detected).  Example: 192.168.0.100

.PARAMETER Wsl2Ip
    WSL2 instance IP (default: auto-detected from wsl hostname -I).
    Run "ifconfig" inside WSL2 to find it.  Example: 172.29.13.112

.PARAMETER WslDistro
    Name of the WSL2 distro (default: default distro).

.PARAMETER StartPort
    First UDP port to relay (default: 7400).

.PARAMETER EndPort
    Last UDP port to relay (default: 7500).

.EXAMPLE
    # Auto-detect IPs (run as Administrator):
    .\windows_wsl2_dds_portforward.ps1

.EXAMPLE
    # Explicit IPs — use these when auto-detect picks the wrong NIC:
    .\windows_wsl2_dds_portforward.ps1 -WindowsLanIp 192.168.0.100 -Wsl2Ip 172.29.13.112

.NOTES
    Must be run as Administrator.
    Re-run after every WSL2 restart (WSL2 IP changes on restart).

    Check running jobs :  Get-Job -Name "DDS_UDP_*"
    Stop all relay jobs:  Get-Job -Name "DDS_UDP_*" | Stop-Job; Get-Job -Name "DDS_UDP_*" | Remove-Job
#>

param(
    [string]$WindowsLanIp = "",
    [string]$Wsl2Ip       = "",
    [string]$WslDistro    = "",
    [int]   $StartPort    = 7400,
    [int]   $EndPort      = 7500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Require Administrator ───────────────────────────────────────────────────────
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator. Re-launch PowerShell with 'Run as Administrator'."
    exit 1
}

# ── WSL distro arg ──────────────────────────────────────────────────────────────
$wslArgs = if ($WslDistro) { @("-d", $WslDistro) } else { @() }

# ── Detect / validate WSL2 IP ──────────────────────────────────────────────────
if ($Wsl2Ip) {
    Write-Host "Using provided WSL2 IP: $Wsl2Ip" -ForegroundColor Cyan
} else {
    Write-Host "Detecting WSL2 IP address..." -ForegroundColor Cyan
    $raw = (wsl @wslArgs -- hostname -I 2>$null)
    if ($raw) { $Wsl2Ip = $raw.Trim().Split(" ")[0] }
}
if (-not $Wsl2Ip) {
    Write-Error "Could not determine WSL2 IP. Pass it explicitly: -Wsl2Ip 172.29.13.112`nRun 'ifconfig' inside WSL2 to find it."
    exit 1
}
Write-Host "  WSL2 IP      : $Wsl2Ip" -ForegroundColor Green

# ── Detect / validate Windows LAN IP ───────────────────────────────────────────
if ($WindowsLanIp) {
    Write-Host "Using provided Windows LAN IP: $WindowsLanIp" -ForegroundColor Cyan
} else {
    Write-Host "Detecting Windows LAN IP..." -ForegroundColor Cyan
    $WindowsLanIp = (
        Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "127.*"     -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.IPAddress -notlike "172.*"     -and
            $_.InterfaceAlias -notlike "*Loopback*" -and
            $_.InterfaceAlias -notlike "*WSL*"
        } |
        Sort-Object PrefixLength |
        Select-Object -First 1
    ).IPAddress
}
if (-not $WindowsLanIp) {
    Write-Error "Could not auto-detect Windows LAN IP. Pass it: -WindowsLanIp 192.168.0.100"
    exit 1
}
Write-Host "  Windows LAN  : $WindowsLanIp" -ForegroundColor Green

# ── Port range info ─────────────────────────────────────────────────────────────
$client3SpdpPort = 7418
$client3DataPort = 7419
Write-Host ""
Write-Host "  CLIENT_ID=3 SPDP port : $client3SpdpPort  (critical for DDS discovery)" -ForegroundColor Yellow
Write-Host "  CLIENT_ID=3 data port : $client3DataPort  (needed for model weights)" -ForegroundColor Yellow
Write-Host "  Relay range           : $StartPort - $EndPort" -ForegroundColor Yellow

# ── Stop existing relay jobs ────────────────────────────────────────────────────
Write-Host "`nStopping any existing DDS relay jobs..." -ForegroundColor Cyan
$existing = Get-Job -Name "DDS_UDP_*" -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Stop-Job
    $existing | Remove-Job -Force
    Write-Host "  Stopped $($existing.Count) existing job(s)." -ForegroundColor Yellow
} else {
    Write-Host "  None running." -ForegroundColor Gray
}

# ── Windows Firewall rule ───────────────────────────────────────────────────────
Write-Host "`nUpdating Windows Firewall rule for DDS UDP $StartPort-$EndPort ..." -ForegroundColor Cyan
$ruleName = "CycloneDDS_WSL2_CLIENT3_UDP"
netsh advfirewall firewall delete rule name=$ruleName 2>$null | Out-Null
netsh advfirewall firewall add rule `
    name=$ruleName `
    dir=in `
    action=allow `
    protocol=UDP `
    localport="${StartPort}-${EndPort}" `
    description="CycloneDDS SPDP/RTPS UDP relay for WSL2 CLIENT_ID=3" | Out-Null
Write-Host "  Firewall rule '$ruleName' added (UDP $StartPort-$EndPort inbound)." -ForegroundColor Green

# ── UDP relay script block ─────────────────────────────────────────────────────
# Each background job:
#   1. Binds a UdpClient on <WindowsLanIp>:<port>  — catches packets from the server
#      and other LAN clients that send to the Windows host IP.
#   2. Forwards every datagram to <Wsl2Ip>:<port>  — delivers it to CycloneDDS
#      running inside WSL2 (Docker --network host).
#   3. Skips datagrams whose source is already the WSL2 IP to prevent relay loops.
#
# Why NOT socat: socat inside WSL2 can only bind to the WSL2 virtual IP
# (172.x.x.x). Packets sent to the Windows LAN IP (192.168.0.100) never reach
# the WSL2 socat process — Windows has no route to forward them. PowerShell
# UdpClient runs on the Windows side and CAN bind to 192.168.0.100 directly.

$relayBlock = {
    param([string]$WinIp, [string]$Wsl2, [int]$Port)
    try {
        $winEp  = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Parse($WinIp), $Port)
        $wsl2Ep = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Parse($Wsl2),  $Port)
        $udp    = [System.Net.Sockets.UdpClient]::new()
        $udp.Client.SetSocketOption(
            [System.Net.Sockets.SocketOptionLevel]::Socket,
            [System.Net.Sockets.SocketOptionName]::ReuseAddress, $true)
        $udp.Client.Bind($winEp)
        $udp.Client.ReceiveTimeout = 500   # ms — allows clean shutdown

        while ($true) {
            try {
                $remoteEp = [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Any, 0)
                $data = $udp.Receive([ref]$remoteEp)
                # Forward only if sender is NOT the WSL2 instance (avoid echo loops)
                if ($remoteEp.Address.ToString() -ne $Wsl2) {
                    $fwd = [System.Net.Sockets.UdpClient]::new()
                    [void]$fwd.Send($data, $data.Length, $wsl2Ep)
                    $fwd.Close()
                }
            } catch [System.Net.Sockets.SocketException] {
                # ReceiveTimeout fired — loop back and wait again
            }
        }
    } finally {
        if ($null -ne $udp) { $udp.Close() }
    }
}

# ── Launch relay jobs ───────────────────────────────────────────────────────────
Write-Host "`nStarting PowerShell UDP relay jobs..." -ForegroundColor Cyan
Write-Host "  ${WindowsLanIp}:<port>  ->  ${Wsl2Ip}:<port>  for ports $StartPort to $EndPort" -ForegroundColor Cyan

$started = 0
for ($port = $StartPort; $port -le $EndPort; $port++) {
    Start-Job -Name "DDS_UDP_$port" `
              -ScriptBlock $relayBlock `
              -ArgumentList $WindowsLanIp, $Wsl2Ip, $port | Out-Null
    $started++
}
Write-Host "  $started relay job(s) started." -ForegroundColor Green

# Give jobs a moment to bind their sockets
Start-Sleep -Milliseconds 800

# ── Verify the two critical ports are bound ────────────────────────────────────
Write-Host "`nVerifying critical ports..." -ForegroundColor Cyan
foreach ($cp in @($client3SpdpPort, $client3DataPort)) {
    $job = Get-Job -Name "DDS_UDP_$cp" -ErrorAction SilentlyContinue
    if ($job -and $job.State -eq "Running") {
        Write-Host "  Port $cp : relay job RUNNING  [OK]" -ForegroundColor Green
    } else {
        Write-Host "  Port $cp : relay job NOT running  [!!] (check if another process holds this port)" -ForegroundColor Red
    }
}

# ── Summary ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  DDS UDP relay is ACTIVE for CLIENT_ID=3 (WSL2)"               -ForegroundColor Cyan
Write-Host "  Windows LAN IP : $WindowsLanIp"                               -ForegroundColor Cyan
Write-Host "  WSL2 IP        : $Wsl2Ip"                                     -ForegroundColor Cyan
Write-Host "  Ports relayed  : $StartPort - $EndPort"                       -ForegroundColor Cyan
Write-Host "  SPDP port 7418 : forwarded  (DDS discovery)"                  -ForegroundColor Cyan
Write-Host "  Data port 7419 : forwarded  (model weight transfer)"          -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "CHECKLIST before starting the client:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [Windows - done] Run this script as Administrator  [OK]" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [WSL2 / distributed_client_gui.py]" -ForegroundColor Yellow
Write-Host "    1. In the GUI 'DDS peer' section:" -ForegroundColor Yellow
Write-Host "         DDS peer - server host   : 192.168.0.102" -ForegroundColor Yellow
Write-Host "         DDS peer - client 1 host : 192.168.0.102" -ForegroundColor Yellow
Write-Host "         DDS peer - client 2 host : 192.168.0.101" -ForegroundColor Yellow
Write-Host "         DDS peer - client 3 host : 192.168.0.100" -ForegroundColor Yellow
Write-Host "    2. Tick   'Running inside WSL2 on Windows (DDS NAT fix)'" -ForegroundColor Yellow
Write-Host "    3. Set    Windows host LAN IP = $WindowsLanIp" -ForegroundColor Yellow
Write-Host "    4. Leave  WSL2 network interface = eth0" -ForegroundColor Yellow
Write-Host "       (confirm with:  ip link  inside WSL2)" -ForegroundColor Yellow
Write-Host "    5. Set    Client ID = 3" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [Ubuntu server - distributed_3pc.env already updated]" -ForegroundColor Yellow
Write-Host "    DDS_PEER_CLIENT2=192.168.0.101  (Linux PC)" -ForegroundColor Yellow
Write-Host "    DDS_PEER_CLIENT3=192.168.0.100  (Windows/WSL2)" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Manage jobs:" -ForegroundColor Cyan
Write-Host "  Check : Get-Job -Name ""DDS_UDP_*""" -ForegroundColor Cyan
Write-Host "  Stop  : Get-Job -Name ""DDS_UDP_*"" | Stop-Job; Get-Job -Name ""DDS_UDP_*"" | Remove-Job" -ForegroundColor Cyan
Write-Host ""
Write-Host "Re-run after every WSL2 restart (WSL2 IP changes each time)." -ForegroundColor Cyan
Write-Host ""
Write-Host "PERMANENT FIX (Windows 11 22H2+):" -ForegroundColor Green
Write-Host "  notepad %USERPROFILE%\.wslconfig" -ForegroundColor Green
Write-Host "  Add:  [wsl2]" -ForegroundColor Green
Write-Host "        networkingMode=mirrored" -ForegroundColor Green
Write-Host "  Then: wsl --shutdown" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
