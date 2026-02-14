"""
Packet Logs Tab - GUI Implementation Guide
For Unified FL Scenario with Server/Client Packet Visualization
"""

import sys
import sqlite3
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QComboBox, QPushButton, QLabel,
                             QGroupBox, QHeaderView, QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor


def _shared_data_dir():
    """Resolve shared_data path (works when GUI runs from project root or Network_Simulation)."""
    # Prefer project root (where Docker mounts shared_data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # GUI/packet_logs_tab.py -> project root
    for base in [project_root, os.getcwd()]:
        path = os.path.join(base, "shared_data")
        if os.path.isdir(path):
            return path
    # Ensure directory exists at project root so Docker mount has a target
    path = os.path.join(project_root, "shared_data")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


class PacketLogsTab(QWidget):
    """
    Packet Logs Visualization Tab
    
    Features:
    - Server/Client packet log selection
    - Separate tables for sent and received packets
    - Protocol filtering
    - Real-time auto-refresh
    - Statistics display
    """
    
    def __init__(self):
        super().__init__()
        self.current_node_type = "server"
        self.current_client_id = None
        self.protocol_filter = "All"
        self._refresh_cycle = 0  # for periodic node list update
        
        self.init_ui()
        self.setup_refresh_timer()
    
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # === Top Controls ===
        controls_layout = QHBoxLayout()
        
        # Node selector (Server / Client 1 / Client 2 / ...)
        controls_layout.addWidget(QLabel("View Logs From:"))
        self.node_selector = QComboBox()
        self.update_node_list()
        self.node_selector.currentTextChanged.connect(self.on_node_changed)
        controls_layout.addWidget(self.node_selector)
        
        # Protocol filter
        controls_layout.addWidget(QLabel("Protocol:"))
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["All", "MQTT", "AMQP", "gRPC", "QUIC", "DDS"])
        self.protocol_combo.currentTextChanged.connect(self.on_protocol_filter_changed)
        controls_layout.addWidget(self.protocol_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        
        # Auto-refresh toggle
        self.auto_refresh_btn = QPushButton("Auto-Refresh: ON")
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.setChecked(True)
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_btn)
        
        self.download_excel_btn = QPushButton("📥 Download to Excel")
        self.download_excel_btn.setStyleSheet("padding: 6px 12px; font-weight: bold;")
        self.download_excel_btn.clicked.connect(self.download_packet_logs_excel)
        controls_layout.addWidget(self.download_excel_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # === Statistics Panel ===
        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout()
        
        self.total_sent_label = QLabel("Sent: 0")
        self.total_received_label = QLabel("Received: 0")
        self.mqtt_count_label = QLabel("MQTT: 0")
        self.amqp_count_label = QLabel("AMQP: 0")
        self.grpc_count_label = QLabel("gRPC: 0")
        self.quic_count_label = QLabel("QUIC: 0")
        self.dds_count_label = QLabel("DDS: 0")
        
        stats_layout.addWidget(self.total_sent_label)
        stats_layout.addWidget(self.total_received_label)
        stats_layout.addWidget(QLabel("|"))
        stats_layout.addWidget(self.mqtt_count_label)
        stats_layout.addWidget(self.amqp_count_label)
        stats_layout.addWidget(self.grpc_count_label)
        stats_layout.addWidget(self.quic_count_label)
        stats_layout.addWidget(self.dds_count_label)
        stats_layout.addStretch()
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # === Sent Packets Table ===
        sent_group = QGroupBox("Sent Packets")
        sent_layout = QVBoxLayout()
        
        self.sent_table = QTableWidget()
        self.sent_table.setColumnCount(7)
        self.sent_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Size (bytes)", "Peer", "Protocol", "Round", "Info"
        ])
        self.sent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sent_table.setAlternatingRowColors(True)
        self.sent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        sent_layout.addWidget(self.sent_table)
        sent_group.setLayout(sent_layout)
        layout.addWidget(sent_group)
        
        # === Received Packets Table ===
        received_group = QGroupBox("Received Packets")
        received_layout = QVBoxLayout()
        
        self.received_table = QTableWidget()
        self.received_table.setColumnCount(7)
        self.received_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Size (bytes)", "Peer", "Protocol", "Round", "Info"
        ])
        self.received_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.received_table.setAlternatingRowColors(True)
        self.received_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        received_layout.addWidget(self.received_table)
        received_group.setLayout(received_layout)
        layout.addWidget(received_group)
        
        self.setLayout(layout)
    
    def update_node_list(self):
        """Update list of available nodes (server + clients)"""
        self.node_selector.clear()
        shared = _shared_data_dir()
        
        # Add server
        if os.path.exists(os.path.join(shared, "packet_logs_server.db")):
            self.node_selector.addItem("Server")
        
        # Add clients (check for client databases)
        for i in range(1, 11):  # Support up to 10 clients
            client_db = os.path.join(shared, f"packet_logs_client_{i}.db")
            if os.path.exists(client_db):
                self.node_selector.addItem(f"Client {i}")
    
    def on_node_changed(self, node_text):
        """Handle node selection change"""
        if node_text.startswith("Client"):
            self.current_node_type = "client"
            self.current_client_id = int(node_text.split()[1])
        else:
            self.current_node_type = "server"
            self.current_client_id = None
        
        self.refresh_data()
    
    def on_protocol_filter_changed(self, protocol):
        """Handle protocol filter change"""
        self.protocol_filter = protocol
        self.refresh_data()
    
    def get_db_path(self):
        """Get database path for current node"""
        shared = _shared_data_dir()
        if self.current_node_type == "server":
            return os.path.join(shared, "packet_logs_server.db")
        else:
            return os.path.join(shared, f"packet_logs_client_{self.current_client_id}.db")
    
    def refresh_data(self):
        """Refresh packet data from database"""
        # Periodically refresh node list so new DBs (after experiment start) appear
        self._refresh_cycle += 1
        if self._refresh_cycle % 5 == 0:
            self.update_node_list()
        
        db_path = self.get_db_path()
        
        if not os.path.exists(db_path):
            # Log only occasionally to avoid terminal spam (e.g. before containers run)
            if not getattr(self, "_last_db_missing_log", None) or self._last_db_missing_log != db_path:
                self._last_db_missing_log = db_path
                print(f"Packet logs: database not found at {db_path} (run unified experiment to create it)")
            return
        self._last_db_missing_log = None
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Build query with protocol filter
            protocol_condition = ""
            if self.protocol_filter != "All":
                protocol_condition = f"WHERE protocol = '{self.protocol_filter}'"
            
            # Query sent packets
            cursor.execute(f"""
                SELECT id, timestamp, packet_size, peer, protocol, round, extra_info
                FROM sent_packets
                {protocol_condition}
                ORDER BY id DESC
                LIMIT 1000
            """)
            sent_packets = cursor.fetchall()
            
            # Query received packets
            cursor.execute(f"""
                SELECT id, timestamp, packet_size, peer, protocol, round, extra_info
                FROM received_packets
                {protocol_condition}
                ORDER BY id DESC
                LIMIT 1000
            """)
            received_packets = cursor.fetchall()
            
            # Update tables
            self.populate_table(self.sent_table, sent_packets)
            self.populate_table(self.received_table, received_packets)
            
            # Update statistics
            self.update_statistics(conn)
            
            conn.close()
            
        except Exception as e:
            print(f"Error refreshing data: {e}")
    
    def populate_table(self, table, data):
        """Populate table with packet data"""
        table.setRowCount(0)
        
        for row_idx, row_data in enumerate(data):
            table.insertRow(row_idx)
            
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if cell_data is not None else "")
                
                # Color code by protocol
                protocol = row_data[4]  # Protocol column
                if protocol == "MQTT":
                    item.setBackground(QColor(230, 255, 230))  # Light green
                elif protocol == "AMQP":
                    item.setBackground(QColor(230, 230, 255))  # Light blue
                elif protocol == "gRPC":
                    item.setBackground(QColor(255, 230, 230))  # Light red
                elif protocol == "QUIC":
                    item.setBackground(QColor(255, 255, 230))  # Light yellow
                elif protocol == "DDS":
                    item.setBackground(QColor(255, 230, 255))  # Light magenta
                
                table.setItem(row_idx, col_idx, item)
    
    def update_statistics(self, conn):
        """Update statistics labels"""
        cursor = conn.cursor()
        
        # Total sent/received
        cursor.execute("SELECT COUNT(*) FROM sent_packets")
        total_sent = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM received_packets")
        total_received = cursor.fetchone()[0]
        
        # Per protocol counts
        protocol_counts = {}
        for protocol in ["MQTT", "AMQP", "gRPC", "QUIC", "DDS"]:
            cursor.execute(f"""
                SELECT COUNT(*) FROM sent_packets WHERE protocol = '{protocol}'
                UNION ALL
                SELECT COUNT(*) FROM received_packets WHERE protocol = '{protocol}'
            """)
            counts = cursor.fetchall()
            protocol_counts[protocol] = sum(c[0] for c in counts)
        
        # Update labels
        self.total_sent_label.setText(f"Sent: {total_sent}")
        self.total_received_label.setText(f"Received: {total_received}")
        self.mqtt_count_label.setText(f"MQTT: {protocol_counts.get('MQTT', 0)}")
        self.amqp_count_label.setText(f"AMQP: {protocol_counts.get('AMQP', 0)}")
        self.grpc_count_label.setText(f"gRPC: {protocol_counts.get('gRPC', 0)}")
        self.quic_count_label.setText(f"QUIC: {protocol_counts.get('QUIC', 0)}")
        self.dds_count_label.setText(f"DDS: {protocol_counts.get('DDS', 0)}")
    
    def setup_refresh_timer(self):
        """Setup auto-refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(3000)  # Refresh every 3 seconds
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off"""
        if self.auto_refresh_btn.isChecked():
            self.refresh_timer.start(3000)
            self.auto_refresh_btn.setText("Auto-Refresh: ON")
        else:
            self.refresh_timer.stop()
            self.auto_refresh_btn.setText("Auto-Refresh: OFF")

    def download_packet_logs_excel(self):
        """Export packet logs from server and all clients to one Excel file with separate sheets."""
        try:
            import pandas as pd
        except ImportError:
            QMessageBox.warning(self, "Export", "pandas is required. Install with: pip install pandas")
            return
        try:
            from openpyxl import __version__  # noqa: F401
        except ImportError:
            QMessageBox.warning(self, "Export", "openpyxl is required for Excel export. Install with: pip install openpyxl")
            return

        shared = _shared_data_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Packet Logs", "", "Excel (*.xlsx);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".xlsx"):
            path = path + ".xlsx"

        sheets = {}
        # Server
        server_db = os.path.join(shared, "packet_logs_server.db")
        if os.path.exists(server_db):
            try:
                conn = sqlite3.connect(server_db)
                sent = pd.read_sql_query("SELECT * FROM sent_packets ORDER BY id", conn)
                recv = pd.read_sql_query("SELECT * FROM received_packets ORDER BY id", conn)
                conn.close()
                sent.insert(0, "Type", "Sent")
                recv.insert(0, "Type", "Received")
                sheets["Server"] = pd.concat([sent, recv], ignore_index=True)
            except Exception as e:
                sheets["Server"] = pd.DataFrame([{"Error": str(e)}])

        # Clients
        for i in range(1, 11):
            client_db = os.path.join(shared, f"packet_logs_client_{i}.db")
            if not os.path.exists(client_db):
                continue
            try:
                conn = sqlite3.connect(client_db)
                sent = pd.read_sql_query("SELECT * FROM sent_packets ORDER BY id", conn)
                recv = pd.read_sql_query("SELECT * FROM received_packets ORDER BY id", conn)
                conn.close()
                sent.insert(0, "Type", "Sent")
                recv.insert(0, "Type", "Received")
                sheets[f"Client {i}"] = pd.concat([sent, recv], ignore_index=True)
            except Exception as e:
                sheets[f"Client {i}"] = pd.DataFrame([{"Error": str(e)}])

        if not sheets:
            QMessageBox.information(self, "Export", "No packet log databases found in shared_data.")
            return

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            QMessageBox.information(self, "Export", f"Saved to:\n{path}\nSheets: {', '.join(sheets.keys())}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# === Integration into Main GUI ===
"""
To integrate into your main GUI:

1. Import the PacketLogsTab class
2. Add it to your tab widget:

from packet_logs_tab import PacketLogsTab

# In your main window __init__:
self.tabs = QTabWidget()
self.packet_logs_tab = PacketLogsTab()
self.tabs.addTab(self.packet_logs_tab, "Packet Logs")

3. Ensure shared_data/ directory exists and is accessible
"""


# === Example Standalone Usage ===
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Packet Logs Viewer")
    window.setGeometry(100, 100, 1200, 800)
    
    packet_logs_tab = PacketLogsTab()
    window.setCentralWidget(packet_logs_tab)
    
    window.show()
    sys.exit(app.exec_())
