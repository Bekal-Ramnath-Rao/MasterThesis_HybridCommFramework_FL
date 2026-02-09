import sqlite3
from tabulate import tabulate

def fetch_packet_logs(db_path='packet_logs.db', table='sent_packets'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f'SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 100')
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    return columns, rows

def display_packet_logs():
    print("\n=== Sent Packets ===")
    sent_cols, sent_rows = fetch_packet_logs('packet_logs.db', 'sent_packets')
    print(tabulate(sent_rows, headers=sent_cols, tablefmt='psql'))
    print("\n=== Received Packets ===")
    recv_cols, recv_rows = fetch_packet_logs('packet_logs.db', 'received_packets')
    print(tabulate(recv_rows, headers=recv_cols, tablefmt='psql'))

if __name__ == "__main__":
    display_packet_logs()
