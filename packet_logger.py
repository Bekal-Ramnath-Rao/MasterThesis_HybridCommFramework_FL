import sqlite3
from datetime import datetime
import os


# Robust DB_PATH logic: Use shared directory for Docker volume mounting
def get_db_path():
    # Docker logic - use /shared_data for volume mounting to local host
    if os.path.exists('/app'):
        # Use shared directory that can be mounted to host
        shared_dir = '/shared_data'
        if not os.path.exists(shared_dir):
            try:
                os.makedirs(shared_dir, exist_ok=True)
            except Exception:
                pass
        
        # Determine if this is server or client based on environment
        node_type = os.getenv('NODE_TYPE', 'unknown')  # Set in docker-compose
        client_id = os.getenv('CLIENT_ID', '')
        
        if node_type == 'server' or 'Server' in os.getcwd():
            candidate = os.path.join(shared_dir, 'packet_logs_server.db')
        elif node_type == 'client' or 'Client' in os.getcwd():
            candidate = os.path.join(shared_dir, f'packet_logs_client_{client_id}.db')
        else:
            candidate = os.path.join(shared_dir, 'packet_logs.db')
    else:
        # Local dev: use project root
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), 'packet_logs.db'))

    # If candidate is a directory, fallback to a safe file in current dir
    if os.path.isdir(candidate):
        import warnings
        warnings.warn(f"DB_PATH {candidate} is a directory! Falling back to ./packet_logs.db")
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), 'packet_logs.db'))
    
    print(f"[PacketLogger] Using database: {candidate}")
    return candidate

DB_PATH = get_db_path()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sent_packets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            packet_size INTEGER NOT NULL,
            peer TEXT NOT NULL,
            protocol TEXT NOT NULL,
            round INTEGER,
            extra_info TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS received_packets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            packet_size INTEGER NOT NULL,
            peer TEXT NOT NULL,
            protocol TEXT NOT NULL,
            round INTEGER,
            extra_info TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_sent_packet(packet_size, peer, protocol, round=None, extra_info=None):
    #print(f"[DEBUG] log_sent_packet: Logging to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sent_packets (timestamp, packet_size, peer, protocol, round, extra_info)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), packet_size, peer, protocol, round, extra_info))
    conn.commit()
    conn.close()

def log_received_packet(packet_size, peer, protocol, round=None, extra_info=None):
    #print(f"[DEBUG] log_received_packet: Logging to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO received_packets (timestamp, packet_size, peer, protocol, round, extra_info)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), packet_size, peer, protocol, round, extra_info))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    # Example usage:
    # log_sent_packet(1024, 'client1', 'DDS', 1, 'test packet')
    # log_received_packet(1024, 'server', 'DDS', 1, 'test packet')
