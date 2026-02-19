"""
Q-Learning Logs Tab - Display Q-learning table for unified FL clients.
Reads from shared_data/q_learning_client_{id}.db (same as packet_logger layout).
"""

import sys
import sqlite3
import os
import glob
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QComboBox, QPushButton, QLabel,
    QGroupBox, QHeaderView, QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor


def _shared_data_dir():
    """Resolve shared_data path (works when GUI runs from project root or Network_Simulation)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    for base in [project_root, os.getcwd()]:
        path = os.path.join(base, "shared_data")
        if os.path.isdir(path):
            return path
    path = os.path.join(project_root, "shared_data")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


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
        self.download_excel_btn = QPushButton("📥 Download to Excel")
        self.download_excel_btn.setStyleSheet("padding: 6px 12px; font-weight: bold;")
        self.download_excel_btn.clicked.connect(self.download_q_learning_excel)
        controls.addWidget(self.download_excel_btn)
        self.reset_q_table_btn = QPushButton("🔄 Reset Q-Table")
        self.reset_q_table_btn.setStyleSheet("padding: 6px 12px; font-weight: bold; background-color: #ffcccc;")
        self.reset_q_table_btn.setToolTip("Reset Q-table to start fresh training. Excel export will be done automatically before reset.")
        self.reset_q_table_btn.clicked.connect(self.reset_q_table_with_backup)
        controls.addWidget(self.reset_q_table_btn)
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
        """Always show Client 1 and Client 2 (and any client with existing DB) so user can select before DBs exist."""
        self.node_selector.clear()
        shared = _shared_data_dir()
        # Always offer Client 1 and Client 2 (typical unified setup), then others if DB exists
        for i in range(1, 11):
            path = os.path.join(shared, f"q_learning_client_{i}.db")
            if i <= 2 or os.path.exists(path):
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
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("—"))
            self.table.setItem(0, 1, QTableWidgetItem("No data yet. Run an RL-unified experiment to populate Q-learning logs for this client."))
            for c in range(2, self.table.columnCount()):
                self.table.setItem(0, c, QTableWidgetItem(""))
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

    def download_q_learning_excel(self):
        """Export Q-learning logs from all clients to one Excel file with separate sheets per client."""
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
            self, "Save Q-Learning Logs", "", "Excel (*.xlsx);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".xlsx"):
            path = path + ".xlsx"

        sheets = {}
        for i in range(1, 11):
            db_path = os.path.join(shared, f"q_learning_client_{i}.db")
            if not os.path.exists(db_path):
                continue
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(
                    """SELECT id, timestamp, round_num, episode, state_network, state_resource,
                       state_model_size, state_mobility, action, reward, q_delta, epsilon,
                       avg_reward_last_100, converged FROM q_learning_log ORDER BY id""",
                    conn
                )
                conn.close()
                sheets[f"Client {i}"] = df
            except Exception as e:
                sheets[f"Client {i}"] = pd.DataFrame([{"Error": str(e)}])

        if not sheets:
            QMessageBox.information(self, "Export", "No Q-learning databases found in shared_data.")
            return

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            QMessageBox.information(self, "Export", f"Saved to:\n{path}\nSheets: {', '.join(sheets.keys())}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def reset_q_table_with_backup(self):
        """Reset Q-table after automatically exporting Excel backup."""
        # First, check if there's any data to backup
        shared = _shared_data_dir()
        has_data = False
        backup_path = None
        
        for i in range(1, 11):
            db_path = os.path.join(shared, f"q_learning_client_{i}.db")
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM q_learning_log")
                    count = cur.fetchone()[0]
                    conn.close()
                    if count > 0:
                        has_data = True
                        break
                except:
                    pass
        
        if has_data:
            # Auto-export Excel before resetting
            reply = QMessageBox.question(
                self,
                "Reset Q-Table",
                "This will reset the Q-table and clear all learned knowledge.\n\n"
                "It will also delete Q-learning database files:\n"
                "• q_learning_*.db\n\n"
                "An Excel backup will be created automatically before reset.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Auto-export to a timestamped file
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(shared, f"q_learning_backup_{timestamp}.xlsx")
                
                # Use the same export logic
                import pandas as pd
                from openpyxl import __version__  # noqa: F401
                
                sheets = {}
                for i in range(1, 11):
                    db_path = os.path.join(shared, f"q_learning_client_{i}.db")
                    if not os.path.exists(db_path):
                        continue
                    try:
                        conn = sqlite3.connect(db_path)
                        df = pd.read_sql_query(
                            """SELECT id, timestamp, round_num, episode, state_network, state_resource,
                               state_model_size, state_mobility, action, reward, q_delta, epsilon,
                               avg_reward_last_100, converged FROM q_learning_log ORDER BY id""",
                            conn
                        )
                        conn.close()
                        if len(df) > 0:
                            sheets[f"Client {i}"] = df
                    except Exception as e:
                        sheets[f"Client {i}"] = pd.DataFrame([{"Error": str(e)}])
                
                if sheets:
                    with pd.ExcelWriter(backup_path, engine="openpyxl") as writer:
                        for name, df in sheets.items():
                            df.to_excel(writer, sheet_name=name[:31], index=False)
                    QMessageBox.information(
                        self,
                        "Backup Created",
                        f"Excel backup saved to:\n{backup_path}\n\nProceeding with Q-table reset..."
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "No Data",
                        "No Q-learning data found to backup. Proceeding with reset anyway..."
                    )
            except Exception as e:
                backup_reply = QMessageBox.warning(
                    self,
                    "Backup Warning",
                    f"Could not create Excel backup: {e}\n\nProceed with reset anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if backup_reply == QMessageBox.No:
                    return
        else:
            # No data to backup, but still ask for confirmation to delete databases
            reply = QMessageBox.question(
                self,
                "Reset Q-Table",
                "This will reset the Q-table and delete Q-learning database files:\n"
                "• q_learning_*.db\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Now reset Q-tables and database files for all clients
        try:
            # Find all Q-table files
            q_table_patterns = [
                os.path.join(shared, "q_table_*.pkl"),
                os.path.join(os.path.dirname(shared), "q_table_*.pkl"),  # Also check parent dir
            ]
            
            # Find all Q-learning database files to delete (only Q-learning, not packet logs)
            db_patterns = [
                os.path.join(shared, 'q_learning_*.db'),
                os.path.join(shared, 'q_learning.db')
            ]
            
            deleted_q_tables = []
            deleted_databases = []
            failed_deletions = []
            
            # Delete Q-table pickle files
            for pattern in q_table_patterns:
                for q_table_path in glob.glob(pattern):
                    try:
                        os.remove(q_table_path)
                        deleted_q_tables.append(os.path.basename(q_table_path))
                        print(f"[Q-Learning] Deleted Q-table: {q_table_path}")
                    except Exception as e:
                        failed_deletions.append((os.path.basename(q_table_path), str(e)))
                        print(f"[Q-Learning] Warning: Could not delete {q_table_path}: {e}")
            
            # Delete database files
            for pattern in db_patterns:
                for db_file in glob.glob(pattern):
                    try:
                        os.remove(db_file)
                        deleted_databases.append(os.path.basename(db_file))
                        print(f"[Q-Learning] Deleted database: {db_file}")
                    except Exception as e:
                        failed_deletions.append((os.path.basename(db_file), str(e)))
                        print(f"[Q-Learning] Warning: Could not delete {db_file}: {e}")
            
            # Show summary message
            total_deleted = len(deleted_q_tables) + len(deleted_databases)
            if total_deleted > 0:
                message = f"Reset complete!\n\n"
                if deleted_q_tables:
                    message += f"Deleted {len(deleted_q_tables)} Q-table file(s):\n"
                    for f in deleted_q_tables[:5]:  # Show first 5
                        message += f"  • {f}\n"
                    if len(deleted_q_tables) > 5:
                        message += f"  ... and {len(deleted_q_tables) - 5} more\n"
                    message += "\n"
                if deleted_databases:
                    message += f"Deleted {len(deleted_databases)} database file(s):\n"
                    for f in deleted_databases[:5]:  # Show first 5
                        message += f"  • {f}\n"
                    if len(deleted_databases) > 5:
                        message += f"  ... and {len(deleted_databases) - 5} more\n"
                    message += "\n"
                message += f"Next experiment will start with fresh Q-table and databases.\n\n"
                if backup_path:
                    message += f"Excel backup saved to:\n{backup_path}"
                
                QMessageBox.information(
                    self,
                    "Reset Complete",
                    message
                )
            else:
                QMessageBox.information(
                    self,
                    "Reset Complete",
                    "No Q-table or database files found to delete.\n"
                    "Next experiment will start with fresh Q-table and databases."
                )
            
            if failed_deletions:
                failed_msg = f"Failed to delete {len(failed_deletions)} file(s):\n"
                for fname, err in failed_deletions[:5]:
                    failed_msg += f"  • {fname}: {err}\n"
                if len(failed_deletions) > 5:
                    failed_msg += f"  ... and {len(failed_deletions) - 5} more\n"
                QMessageBox.warning(
                    self,
                    "Some Files Failed",
                    failed_msg
                )
            
            # Refresh the display
            self.refresh_data()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Reset Error",
                f"Error resetting Q-table and databases: {e}"
            )


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Q-Learning Logs")
    win.setCentralWidget(QLearningLogsTab())
    win.show()
    sys.exit(app.exec_())
