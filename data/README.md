# Data Directory

This directory contains runtime data files and databases.

## Files

### Databases
- `packet_logs.db` - SQLite database containing packet transmission logs for all protocols

## Usage

### Viewing Packet Logs
To view packet logs from the database:
```bash
python scripts/utilities/display_packet_logs.py
```

Or use the GUI:
```bash
python Network_Simulation/experiment_gui.py
```

### Database Schema
The packet logs database contains:
- Protocol-specific transmission data
- Timestamps and latencies
- Payload sizes
- Success/failure status
- Network conditions

## Notes

- The `packet_logs.db` file is automatically created when packet logging is enabled
- The database may grow large during experiments; consider archiving or cleaning periodically
- See `shared_data/` directory for additional database files used by the server
