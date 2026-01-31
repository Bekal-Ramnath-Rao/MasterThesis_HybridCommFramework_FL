import sqlite3
from datetime import datetime

DB_PATH = 'packet_logs.db'  # Use a unique path per client/server if needed

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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sent_packets (timestamp, packet_size, peer, protocol, round, extra_info)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), packet_size, peer, protocol, round, extra_info))
    conn.commit()
    conn.close()

def log_received_packet(packet_size, peer, protocol, round=None, extra_info=None):
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
