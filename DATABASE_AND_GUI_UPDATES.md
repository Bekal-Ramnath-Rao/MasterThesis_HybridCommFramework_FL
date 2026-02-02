# Why No .db Files in shared_data? + GUI Updates

## Issue 1: No .db Files in shared_data Directory

### The Reason
**The database files are NOT created until the Docker containers actually run!**

The packet logger code only executes inside the Docker containers. Here's what happens:

```
1. You start Docker containers → docker-compose up
2. Python code runs INSIDE containers
3. packet_logger.py detects NODE_TYPE environment variable
4. Creates database files in /shared_data (inside container)
5. /shared_data is mounted to ./shared_data on your host
6. You can now see the .db files on your host machine
```

### How to Create the Databases

**Option 1: Run Unified Scenario (RL-based protocol selection)**
```bash
cd Docker
docker-compose -f docker-compose-unified-emotion.yml up
```

Wait a few seconds, then check:
```bash
ls -lh ../shared_data/
```

You should see:
```
packet_logs_server.db
packet_logs_client_1.db
packet_logs_client_2.db
```

**Option 2: Use the GUI**
1. Open the GUI: `cd Network_Simulation && python3 experiment_gui.py`
2. Go to "Basic Config" tab
3. Check "🤖 RL-Unified (Dynamic Selection)" protocol
4. Select network scenario (e.g., "Excellent")
5. Click "▶️ Start Experiment"
6. Wait for containers to start
7. Check `shared_data/` directory

**Option 3: Test Script**
```bash
# This simulates what happens inside containers
python3 test_packet_logger.py
```

### When Will Databases Have Data?

The databases are created immediately when containers start, BUT they will be **empty** until:

1. **Server starts** → Creates `packet_logs_server.db`
2. **Clients register** → Start logging registration packets
3. **Training begins** → Logs model updates, aggregations, etc.
4. **Protocols communicate** → Packets logged for each protocol used

### Verify Database Creation

```bash
# 1. Start containers
cd Docker
docker-compose -f docker-compose-unified-emotion.yml up -d

# 2. Wait 5 seconds for initialization
sleep 5

# 3. Check databases exist
ls -lh ../shared_data/

# 4. Check database contents (should have tables but maybe no rows yet)
sqlite3 ../shared_data/packet_logs_server.db "SELECT COUNT(*) FROM sent_packets;"

# 5. Wait for training to start, then check again
sleep 30
sqlite3 ../shared_data/packet_logs_server.db "SELECT * FROM sent_packets LIMIT 5;"
```

### View Packets in GUI

1. Make sure Docker containers are running
2. Open GUI: `python3 Network_Simulation/experiment_gui.py`
3. Go to **"Packet Logs"** tab
4. Select node: Server / Client 1 / Client 2
5. Watch real-time packets!

---

## Issue 2: GUI Docker Build Tab Updates

### What Was Added

✅ **Three new build buttons for RL-Unified scenarios:**

1. 🤖 Build Unified Images (Temperature Regulation)
2. 🤖 Build Unified Images (Mental State)  
3. 🤖 Build Unified Images (Emotion)

These buttons build Docker images using the unified docker-compose files that include:
- FL_Server_Unified.py (handles all 5 protocols simultaneously)
- FL_Client_Unified.py (uses RL to select best protocol)
- All protocol brokers (MQTT, AMQP, gRPC, QUIC, DDS)

### GUI Changes Made

**File:** `Network_Simulation/experiment_gui.py`

**Location:** Docker Build tab

**Before:**
```
🐳 Docker Image Build Options
├── Build Docker Images (Temperature Regulation)
├── Build Docker Images (Mental State)
└── Build Docker Images (Emotion)
```

**After:**
```
🐳 Docker Image Build Options
├── Build Docker Images (Temperature Regulation)
├── Build Docker Images (Mental State)
├── Build Docker Images (Emotion)
├── ──────────────────────────────────────────
├── 🤖 RL-Unified Scenario (Dynamic Protocol Selection)
├── 🤖 Build Unified Images (Temperature Regulation)
├── 🤖 Build Unified Images (Mental State)
└── 🤖 Build Unified Images (Emotion)
```

### How to Use

1. **Open GUI:**
   ```bash
   cd Network_Simulation
   python3 experiment_gui.py
   ```

2. **Go to "🐳 Docker Build" tab**

3. **Choose build type:**
   - **Single Protocol Builds** (top buttons) → For running experiments with one protocol at a time
   - **Unified Builds** (bottom buttons 🤖) → For RL-based dynamic protocol selection

4. **Click the appropriate button:**
   - e.g., "🤖 Build Unified Images (Emotion)"
   - Build progress shows in the output log
   - Button is disabled during build
   - Re-enabled when complete

### What Gets Built

**Single Protocol Builds:**
- Uses: `docker-compose-emotion.gpu-isolated.yml`
- Includes: One server + clients for ONE specific protocol
- For: Running experiments comparing individual protocols

**Unified Builds (🤖):**
- Uses: `docker-compose-unified-emotion.yml`
- Includes: 
  - FL_Server_Unified.py (all 5 protocol servers running concurrently)
  - FL_Client_Unified.py (RL selector chooses best protocol)
  - MQTT broker (always for control signals)
  - AMQP broker (RabbitMQ)
  - gRPC server
  - QUIC server
  - DDS domain participant
- For: RL-based adaptive protocol selection experiments

### Running Unified Experiments

After building unified images:

**Method 1: GUI**
1. Go to "Basic Config" tab
2. Check "🤖 RL-Unified (Dynamic Selection)" under protocols
3. Select network scenario
4. Click "▶️ Start Experiment"

**Method 2: Command Line**
```bash
cd Network_Simulation
python3 run_network_experiments.py \
  --use-case emotion \
  --protocols rl_unified \
  --scenarios excellent good moderate \
  --rounds 10 \
  --enable-gpu
```

**Method 3: Direct Docker**
```bash
cd Docker
docker-compose -f docker-compose-unified-emotion.yml up
```

### Build Options

Both single and unified builds support:

- **✓ Use Cache When Building Docker Images** (default: checked)
  - Faster builds, uses cached layers
  - Uncheck for clean rebuild

- **✓ Rebuild Docker Images Before Experiment** 
  - Auto-rebuilds before starting experiment
  - Ensures latest code changes are included

### Troubleshooting

**Build fails:**
```bash
# Check Docker is running
docker ps

# Check compose file exists
ls -lh Docker/docker-compose-unified-emotion.yml

# Try manual build
cd Docker
docker-compose -f docker-compose-unified-emotion.yml build --no-cache
```

**Wrong images built:**
- Make sure you clicked the correct button
- Unified buttons (🤖) are at the bottom with red background
- Single protocol buttons are at the top with blue background

**Can't see build output:**
- Build output is shown in the "Docker Build Log" tab
- GUI automatically switches to this tab during build
- If build finishes quickly, check the status bar at bottom

---

## Architecture: How It All Works Together

```
┌─────────────────────────────────────────────────────────────┐
│                     experiment_gui.py                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Docker Build Tab                                      │   │
│  │  - Build Single Protocol Images (blue buttons)       │   │
│  │  - Build Unified Images (red buttons 🤖)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Runs: docker-compose -f docker-compose-unified-*.yml build │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Docker Containers Start                         │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ FL Server        │  │ FL Client 1  │  │ FL Client 2  │  │
│  │ NODE_TYPE=server │  │ NODE_TYPE=   │  │ NODE_TYPE=   │  │
│  │                  │  │ client       │  │ client       │  │
│  │ Creates:         │  │ CLIENT_ID=1  │  │ CLIENT_ID=2  │  │
│  │ packet_logs_     │  │              │  │              │  │
│  │ server.db        │  │ Creates:     │  │ Creates:     │  │
│  │                  │  │ packet_logs_ │  │ packet_logs_ │  │
│  │                  │  │ client_1.db  │  │ client_2.db  │  │
│  └──────────────────┘  └──────────────┘  └──────────────┘  │
│           ↓                    ↓                  ↓         │
│  Volume Mount: /shared_data → ../shared_data (host)        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Host: ./shared_data/ directory                  │
│                                                              │
│  packet_logs_server.db                                      │
│  packet_logs_client_1.db                                    │
│  packet_logs_client_2.db                                    │
│                                                              │
│                          ↑                                   │
│              experiment_gui.py reads these                   │
│              Packet Logs Tab displays them                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Create Databases
```bash
# Option 1: Docker Compose
cd Docker && docker-compose -f docker-compose-unified-emotion.yml up -d

# Option 2: GUI
python3 Network_Simulation/experiment_gui.py
# → Select RL-Unified protocol → Start Experiment

# Option 3: Test locally (simulation)
python3 test_packet_logger.py
```

### Verify Databases
```bash
# List files
ls -lh shared_data/

# Check table structure
sqlite3 shared_data/packet_logs_server.db ".schema"

# Count packets
sqlite3 shared_data/packet_logs_server.db "SELECT COUNT(*) FROM sent_packets;"

# View recent packets
sqlite3 shared_data/packet_logs_server.db "SELECT * FROM sent_packets ORDER BY timestamp DESC LIMIT 10;"
```

### Build Images
```bash
# GUI: Docker Build tab → Click appropriate button

# Or command line:
cd Docker
docker-compose -f docker-compose-unified-emotion.yml build
```

### View Packets
```bash
# GUI: Packet Logs tab → Select node → Auto-refresh enabled

# Or command line:
python3 display_packet_logs.py
```

---

## Summary

✅ **Database Issue Resolved:**
- Databases are created by Docker containers, not manually
- Must run containers first: `docker-compose up`
- Files appear in `shared_data/` on host via volume mount

✅ **GUI Updates Complete:**
- Added 3 unified scenario build buttons (🤖)
- Separated single protocol vs unified builds visually
- All buttons use threaded builds with live output
- Proper error handling and status updates

✅ **Ready to Use:**
1. Build unified images: GUI → Docker Build tab → Click 🤖 button
2. Start containers: GUI → Basic Config → Check RL-Unified → Start
3. View packets: GUI → Packet Logs tab → Select node
4. Watch real-time protocol selection and packet logging!
