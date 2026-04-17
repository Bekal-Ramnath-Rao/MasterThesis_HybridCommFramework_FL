<#
.SYNOPSIS
    Forward DDS/CycloneDDS UDP ports from the Windows host NIC into the WSL2 instance.

.DESCRIPTION
    WSL2 uses a virtual NAT network.  Docker containers started with --network host
    inside WSL2 bind to the WSL2 virtual IP (e.g. 172.x.x.x), NOT to the Windows host
    LAN IP (e.g. 192.168.0.100).  This means:

      1. CycloneDDS advertises the unreachable 172.x.x.x address as its RTPS locator.
         Fix → set  DDS_EXTERNAL_NETWORK_ADDRESS=<Windows-LAN-IP>  so Cyclone
         advertises a reachable address (the distributed_client_gui.py WSL2 mode does
         this automatically).

      2. SPDP discovery packets sent to <Windows-LAN-IP>:7412-7418 by the server and
         other clients arrive at the Windows NIC but are never forwarded to WSL2.
         Fix → this script sets up socat-based UDP relays inside WSL2.

    REQUIREMENTS
    ────────────
    • WSL2 with the distro you use for Docker  (Ubuntu, Debian, …)
    • socat installed inside that distro  →  sudo apt install -y socat
    • This script must be run as Administrator in PowerShell on the Windows host.

    ALTERNATIVE (Windows 11 22H2+, recommended)
    ────────────────────────────────────────────
    Enable WSL2 mirrored networking in  %USERPROFILE%\.wslconfig :

        [wsl2]
        networkingMode=mirrored

    Then restart WSL2:  wsl --shutdown
    With mirrored networking WSL2 shares the Windows host's network stack, so the
    container sees 192.168.0.100 directly — no port-forwarding script needed.

.PARAMETER WslDistro
    Name of the WSL2 distro to use (default: default WSL distro).

.PARAMETER DdsPorts
    Comma-separated list of UDP ports to relay.  Defaults to the SPDP ports for
    participants 1-4 (server=7412, client1=7414, client2=7416, client3=7418) plus
    the base RTPS discovery port 7400 and the RTPS user-data range 7401-7430.
    Widen the range if you see "unreachable locator" warnings in Cyclone logs.

.EXAMPLE
    # Relay default DDS ports into the default WSL2 distro (run as Administrator):
    .\windows_wsl2_dds_portforward.ps1

.EXAMPLE
    # Specify a distro and only the SPDP ports:
    .\windows_wsl2_dds_portforward.ps1 -WslDistro Ubuntu-22.04 -DdsPorts "7412,7414,7416,7418"

.NOTES
    To stop all relays:  wsl -d <distro> -- pkill socat
    To list relays:      wsl -d <distro> -- pgrep -a socat
#>

param(
    [string]$WslDistro   = "",
    [string]$DdsPorts    = "7400,7401,7402,7403,7404,7405,7406,7407,7408,7409,7410,7411,7412,7413,7414,7415,7416,7417,7418,7419,7420,7421,7422,7423,7424,7425,7426,7427,7428,7429,7430"
)

# ── Require Administrator ───────────────────────────────────────────────────────
$currentPrincipal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator. Re-launch PowerShell with 'Run as Administrator'."
    exit 1
}

# ── Resolve WSL2 distro ─────────────────────────────────────────────────────────
$wslArgs = if ($WslDistro) { @("-d", $WslDistro) } else { @() }

# ── Detect WSL2 IP ──────────────────────────────────────────────────────────────
Write-Host "Detecting WSL2 IP address..." -ForegroundColor Cyan
$wsl2Ip = (wsl @wslArgs -- hostname -I 2>$null).Trim().Split(" ")[0]
if (-not $wsl2Ip) {
    Write-Error "Could not determine WSL2 IP.  Is WSL2 running?  Try: wsl --list --running"
    exit 1
}
Write-Host "  WSL2 IP: $wsl2Ip" -ForegroundColor Green

# ── Detect Windows host LAN IP (first non-loopback, non-WSL IPv4) ──────────────
$windowsIp = (
    Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "172.*" -and $_.InterfaceAlias -notlike "*WSL*" -and $_.InterfaceAlias -notlike "*Loopback*" } |
    Sort-Object -Property PrefixLength |
    Select-Object -First 1
).IPAddress
Write-Host "  Windows LAN IP: $windowsIp" -ForegroundColor Green

# ── Check / install socat ───────────────────────────────────────────────────────
Write-Host "`nChecking socat inside WSL2..." -ForegroundColor Cyan
$socatCheck = wsl @wslArgs -- which socat 2>$null
if (-not $socatCheck) {
    Write-Host "  socat not found — installing..." -ForegroundColor Yellow
    wsl @wslArgs -- sudo apt-get install -y socat
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install socat inside WSL2.  Install it manually: sudo apt install -y socat"
        exit 1
    }
}
Write-Host "  socat OK." -ForegroundColor Green

# ── Kill any existing socat DDS relays ─────────────────────────────────────────
Write-Host "`nStopping any existing DDS socat relays in WSL2..." -ForegroundColor Cyan
wsl @wslArgs -- bash -c "pkill -f 'socat.*UDP' 2>/dev/null; echo done" | Out-Null

# ── Open Windows Firewall for incoming DDS UDP ─────────────────────────────────
$portList = $DdsPorts -replace ",", ","    # already comma-separated
Write-Host "`nAdding Windows Firewall rule for DDS UDP ports ($portList)..." -ForegroundColor Cyan
$ruleName = "CycloneDDS_WSL2_UDP"
netsh advfirewall firewall delete rule name=$ruleName | Out-Null
netsh advfirewall firewall add rule `
    name=$ruleName `
    dir=in `
    action=allow `
    protocol=UDP `
    localport=$portList `
    description="Allow CycloneDDS SPDP/RTPS UDP traffic for WSL2 port-forwarding" | Out-Null
Write-Host "  Firewall rule '$ruleName' added." -ForegroundColor Green

# ── Start socat relays inside WSL2 ─────────────────────────────────────────────
Write-Host "`nStarting UDP relays: Windows $windowsIp → WSL2 $wsl2Ip" -ForegroundColor Cyan
$ports = $DdsPorts.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

foreach ($port in $ports) {
    $socatCmd = "nohup socat UDP4-RECVFROM:${port},bind=${wsl2Ip},reuseaddr,fork UDP4-SENDTO:localhost:${port} </dev/null >/tmp/socat_dds_${port}.log 2>&1 &"
    wsl @wslArgs -- bash -c $socatCmd
}
Write-Host "  Started $($ports.Count) UDP relays." -ForegroundColor Green

# ── netsh portproxy note (TCP only, for reference) ─────────────────────────────
Write-Host @"

NOTE: Windows 'netsh interface portproxy' only supports TCP — DDS uses UDP.
The socat relays above handle UDP forwarding.

To verify relays are running inside WSL2:
  wsl $(@($wslArgs) -join ' ') -- pgrep -a socat

To stop all relays:
  wsl $(@($wslArgs) -join ' ') -- pkill socat

ALTERNATIVE (Windows 11 22H2+ — no script needed):
  Add the following to %USERPROFILE%\.wslconfig and run 'wsl --shutdown':

    [wsl2]
    networkingMode=mirrored

  Mirrored networking gives WSL2 the same IP as the Windows host, so DDS
  discovery works across the LAN without any port-forwarding.
"@ -ForegroundColor Cyan

Write-Host "`nDone. CycloneDDS UDP relay is active." -ForegroundColor Green
