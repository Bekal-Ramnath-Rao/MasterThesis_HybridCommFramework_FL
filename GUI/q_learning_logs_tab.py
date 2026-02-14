"""
Q-Learning Logs Tab - Display Q-learning table for unified FL clients.
Reads from shared_data/q_learning_client_{id}.db (same as packet_logger layout).
"""

import sys
import sqlite3
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QComboBox, QPushButton, QLabel,
    QGroupBox, QHeaderView)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor


def _shared_data_dir():
    """Resolve shared_data path (works from GUI or Network_Simulation)."""
    cwd = os.getcwd()
    for base in [cwd, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        path = os.path.join(base, "shared_data")
        if os.path.isdir(path):
            return path
    return os.path.join(cwd, "shared_data")


class QLearningLogsTab(QWidget):
    """
    Q-Learning log visualization: one table with all columns from q_learning_log.
    Node selector: Client 1, Client 2, ... (only clients have Q-learning DBs).
    """

    def __init__(self):
        super().__init__()
        self.current_client_id = 1
        self.init_ui()
        self.setup_refresh_timer()

    def init_ui(self):
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("View Q-Learning From:"))
        self.node_selector = QComboBox()
        self.update_node_list()
        self.node_selector.currentTextChanged.connect(self.on_node_changed)
        controls.addWidget(self.node_selector)
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.refresh_data)
        controls.addWidget(self.refresh_btn)
        self.auto_refresh_btn = QPushButton("Auto-Refresh: ON")
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.setChecked(True)
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        controls.addWidget(self.auto_refresh_btn)
        controls.addStretch()
        layout.addLayout(controls)

        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout()
        self.rows_label = QLabel("Rows: 0")
        self.episodes_label = QLabel("Episodes: 0")
        self.converged_label = QLabel("Converged: No")
        stats_layout.addWidget(self.rows_label)
        stats_layout.addWidget(self.episodes_label)
        stats_layout.addWidget(self.converged_label)
        stats_layout.addStretch()
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Round", "Episode", "State (net)", "State (res)", "State (size)", "State (mob)",
            "Action", "Reward", "Q Delta", "Epsilon", "Avg R(100)", "Converged"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_node_list(self):
        self.node_selector.clear()
        shared = _shared_data_dir()
        for i in range(1, 11):
            path = os.path.join(shared, f"q_learning_client_{i}.db")
            if os.path.exists(path):
                self.node_selector.addItem(f"Client {i}")

    def on_node_changed(self, text):
        if text.startswith("Client"):
            self.current_client_id = int(text.split()[1])
        self.refresh_data()

    def get_db_path(self):
        return os.path.join(_shared_data_dir(), f"q_learning_client_{self.current_client_id}.db")

    def refresh_data(self):
        db_path = self.get_db_path()
        if not os.path.exists(db_path):
            self.table.setRowCount(0)
            self.rows_label.setText("Rows: 0")
            self.episodes_label.setText("Episodes: 0")
            self.converged_label.setText("Converged: No")
            return
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT id, timestamp, round_num, episode, state_network, state_resource,
                       state_model_size, state_mobility, action, reward, q_delta, epsilon,
                       avg_reward_last_100, converged
                FROM q_learning_log
                ORDER BY id ASC
            """)
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            print(f"Q-Learning tab error: {e}")
            return
        self.table.setRowCount(0)
        for r in rows:
            self.table.insertRow(self.table.rowCount())
            for c, val in enumerate(r):
                item = QTableWidgetItem(str(val) if val is not None else "")
                if c == 13 and val == 1:
                    item.setBackground(QColor(200, 255, 200))
                self.table.setItem(self.table.rowCount() - 1, c, item)
        max_ep = max((r[3] for r in rows), default=0)
        converged = any(r[13] == 1 for r in rows)
        self.rows_label.setText(f"Rows: {len(rows)}")
        self.episodes_label.setText(f"Episodes: {max_ep + 1}")
        self.converged_label.setText("Converged: Yes" if converged else "Converged: No")

    def setup_refresh_timer(self):
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(3000)

    def toggle_auto_refresh(self):
        if self.auto_refresh_btn.isChecked():
            self.refresh_timer.start(3000)
            self.auto_refresh_btn.setText("Auto-Refresh: ON")
        else:
            self.refresh_timer.stop()
            self.auto_refresh_btn.setText("Auto-Refresh: OFF")


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Q-Learning Logs")
    win.setCentralWidget(QLearningLogsTab())
    win.show()
    sys.exit(app.exec_())
