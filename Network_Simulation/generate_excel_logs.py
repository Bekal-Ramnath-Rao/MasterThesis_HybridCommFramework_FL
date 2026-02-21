#!/usr/bin/env python3
"""
Generate Excel workbook from pcap files in a completed FL experiment folder.

Looks for server and client pcap files (server.pcap, client1.pcap, client2.pcap
or names containing 'server'/'client'), runs tshark to extract fields into CSV,
then writes a single Excel file with one sheet per pcap (Server, Client1, Client2).

Usage:
  python3 generate_excel_logs.py [experiment_folder]
  python3 generate_excel_logs.py -o /path/to/output.xlsx experiment_results/emotion_20260128_143000/mqtt_poor

  If experiment_folder is omitted, uses the latest modified folder under
  experiment_results/ that contains pcap files (or its latest subfolder with pcaps).

Pip requirements (see requirements-excel-logs.txt):
  pip install pandas openpyxl

System requirement: tshark (Wireshark CLI), e.g. apt-get install tshark
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def find_pcap_folder(base_path: Path) -> Path:
    """
    Resolve the directory that contains pcap files.
    If base_path has .pcap files directly, return it.
    Otherwise find the most recently modified subdirectory that contains .pcap files.
    """
    base_path = Path(base_path).resolve()
    if not base_path.is_dir():
        return base_path

    pcaps_here = list(base_path.glob("*.pcap"))
    if pcaps_here:
        return base_path

    subdirs_with_pcaps = [
        d for d in base_path.iterdir()
        if d.is_dir() and list(d.glob("*.pcap"))
    ]
    if not subdirs_with_pcaps:
        return base_path
    return max(subdirs_with_pcaps, key=lambda d: d.stat().st_mtime)


def find_server_client_pcaps(folder: Path):
    """Find server and client pcap files by name or pattern. Returns (server_path, client1_path, client2_path)."""
    folder = Path(folder)
    all_pcaps = list(folder.glob("*.pcap"))
    if not all_pcaps:
        return None, None, None

    server_path = None
    client1_path = None
    client2_path = None

    # Exact names first
    for p in all_pcaps:
        if p.name == "server.pcap":
            server_path = p
            break
    if not server_path:
        for p in all_pcaps:
            if "server" in p.name.lower():
                server_path = p
                break

    client_pcaps = sorted([p for p in all_pcaps if "client" in p.name.lower()])
    for p in all_pcaps:
        if p.name == "client1.pcap":
            client1_path = p
            break
    if not client1_path and len(client_pcaps) >= 1:
        client1_path = client_pcaps[0]
    for p in all_pcaps:
        if p.name == "client2.pcap":
            client2_path = p
            break
    if not client2_path and len(client_pcaps) >= 2:
        client2_path = client_pcaps[1]

    return server_path, client1_path, client2_path


def run_tshark(pcap_path: Path, csv_path: Path) -> bool:
    """Run tshark on pcap_path and write CSV to csv_path. Returns True on success."""
    fields = [
        "frame.number",
        "frame.time",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "frame.len",
    ]
    cmd = [
        "tshark",
        "-r", str(pcap_path),
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "occurrence=f",
    ]
    for f in fields:
        cmd.extend(["-e", f])
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            cwd=str(pcap_path.parent),
        )
        if result.returncode != 0 and result.stderr and "empty" not in result.stderr.lower():
            return False
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(result.stdout or "")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def pcap_to_dataframe(pcap_path: Path, temp_dir: Path):
    """Run tshark on pcap_path, read CSV into a pandas DataFrame. Returns None on error."""
    import pandas as pd
    csv_path = temp_dir / f"{pcap_path.stem}.csv"
    if not run_tshark(pcap_path, csv_path):
        return None
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate Excel workbook from pcap files in an FL experiment folder."
    )
    parser.add_argument(
        "experiment_folder",
        nargs="?",
        default=None,
        help="Path to experiment result folder (e.g. experiment_results/emotion_20260128_143000 or a protocol_scenario subfolder). If omitted, uses latest folder with pcaps.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output Excel path. Default: fl_network_metrics_<foldername>.xlsx in the same folder.",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. Install with: pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("ERROR: openpyxl is required for .xlsx. Install with: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    experiment_results = project_root / "experiment_results"
    if not experiment_results.is_dir():
        experiment_results = script_dir / "experiment_results"
    if not experiment_results.is_dir():
        print("ERROR: experiment_results directory not found.", file=sys.stderr)
        sys.exit(1)

    if args.experiment_folder:
        base_path = Path(args.experiment_folder).resolve()
        if not base_path.is_absolute():
            base_path = (project_root / base_path).resolve()
        if not base_path.is_dir():
            print(f"ERROR: Folder not found: {base_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Find latest folder under experiment_results that (or whose subdir) has pcaps
        candidates = []
        for d in experiment_results.iterdir():
            if not d.is_dir():
                continue
            folder = find_pcap_folder(d)
            if list(Path(folder).glob("*.pcap")):
                candidates.append((folder, folder.stat().st_mtime))
        if not candidates:
            print("ERROR: No experiment folder with pcap files found under experiment_results.", file=sys.stderr)
            sys.exit(1)
        base_path = max(candidates, key=lambda x: x[1])[0]

    folder = find_pcap_folder(base_path)
    server_path, client1_path, client2_path = find_server_client_pcaps(folder)
    if not server_path and not client1_path and not client2_path:
        print(f"ERROR: No server/client pcap files found in {folder}. Expected server.pcap, client1.pcap, client2.pcap (or names containing 'server'/'client').", file=sys.stderr)
        sys.exit(1)

    # Sanitize sheet name: Excel sheet name max 31 chars, no : \\ / ? * [ ]
    def sheet_name(name: str) -> str:
        s = name.replace(":", "_").replace("\\", "_").replace("/", "_").replace("?", "_").replace("*", "_").replace("[", "_").replace("]", "_")
        return s[:31] if len(s) > 31 else s

    with tempfile.TemporaryDirectory(prefix="fl_excel_logs_") as temp_dir:
        temp_dir = Path(temp_dir)
        dfs = {}
        if server_path and server_path.exists():
            df = pcap_to_dataframe(server_path, temp_dir)
            if df is not None and not df.empty:
                dfs["Server"] = df
            elif server_path.exists() and server_path.stat().st_size == 0:
                print(f"WARNING: {server_path.name} is empty; skipping.", file=sys.stderr)
            else:
                print(f"WARNING: Could not extract data from {server_path.name}; skipping.", file=sys.stderr)
        else:
            if server_path and not server_path.exists():
                print(f"WARNING: server pcap not found; skipping.", file=sys.stderr)

        if client1_path and client1_path.exists():
            df = pcap_to_dataframe(client1_path, temp_dir)
            if df is not None and not df.empty:
                dfs["Client1"] = df
            elif client1_path.stat().st_size == 0:
                print(f"WARNING: {client1_path.name} is empty; skipping.", file=sys.stderr)
            else:
                print(f"WARNING: Could not extract data from {client1_path.name}; skipping.", file=sys.stderr)
        else:
            if client1_path and not client1_path.exists():
                print(f"WARNING: client1 pcap not found; skipping.", file=sys.stderr)

        if client2_path and client2_path.exists():
            df = pcap_to_dataframe(client2_path, temp_dir)
            if df is not None and not df.empty:
                dfs["Client2"] = df
            elif client2_path.stat().st_size == 0:
                print(f"WARNING: {client2_path.name} is empty; skipping.", file=sys.stderr)
            else:
                print(f"WARNING: Could not extract data from {client2_path.name}; skipping.", file=sys.stderr)
        else:
            if client2_path and not client2_path.exists():
                print(f"WARNING: client2 pcap not found; skipping.", file=sys.stderr)

        if not dfs:
            print("ERROR: No valid pcap data could be extracted. Check that tshark is installed and pcaps are not empty.", file=sys.stderr)
            sys.exit(1)

        # Output path
        experiment_folder_name = folder.name
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in experiment_folder_name)
        if args.output:
            out_path = Path(args.output).resolve()
        else:
            out_path = folder / f"fl_network_metrics_{safe_name}.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name(name), index=False)
        print(f"Saved: {out_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
