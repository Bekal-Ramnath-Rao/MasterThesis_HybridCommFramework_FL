#!/usr/bin/env python3
"""
Distributed Client GUI for Federated Learning
Allows running clients on separate PCs that connect to a central experiment server
"""

import sys
import os
import json
import re
import subprocess
import threading
import glob
import shutil
from datetime import datetime

# Outgoing interface in `ip -4 route get` / `ip route show default` output.
_IP_ROUTE_DEV_RE = re.compile(r"\bdev\s+(\S+)")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QSlider,
    QGroupBox, QTextEdit, QProgressBar, QGridLayout,
    QMessageBox, QLineEdit, QFrame, QRadioButton, QButtonGroup,
    QScrollArea, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

NETWORK_SCENARIO_PRESETS = {
    "none": {"latency": 0, "bandwidth": 100, "jitter": 0, "packet_loss": 0.0},
    "excellent": {"latency": 5, "bandwidth": 100, "jitter": 0, "packet_loss": 0.0},
    "good": {"latency": 20, "bandwidth": 50, "jitter": 0, "packet_loss": 0.1},
    "moderate": {"latency": 50, "bandwidth": 20, "jitter": 0, "packet_loss": 0.5},
    "poor": {"latency": 100, "bandwidth": 5, "jitter": 0, "packet_loss": 1.0},
    "very_poor": {"latency": 200, "bandwidth": 1, "jitter": 0, "packet_loss": 3.0},
    "satellite": {"latency": 600, "bandwidth": 10, "jitter": 50, "packet_loss": 2.0},
    "congested_light": {"latency": 30, "bandwidth": 30, "jitter": 5, "packet_loss": 0.5},
    "congested_moderate": {"latency": 75, "bandwidth": 15, "jitter": 15, "packet_loss": 1.5},
    "congested_heavy": {"latency": 150, "bandwidth": 5, "jitter": 30, "packet_loss": 3.0},
}

CONGESTION_PRESETS = {
    "light": {"latency": 10, "jitter": 5, "packet_loss": 0.2, "bandwidth_factor": 0.85},
    "moderate": {"latency": 25, "jitter": 10, "packet_loss": 0.5, "bandwidth_factor": 0.70},
    "heavy": {"latency": 50, "jitter": 20, "packet_loss": 1.0, "bandwidth_factor": 0.50},
}

# Dynamic preset pool / draw logic: network_simulator.draw_dynamic_base_scenario()


class ClientMonitor(QThread):
    """Background thread for monitoring client container"""
    log_update = pyqtSignal(str)
    
    def __init__(self, container_name, parent=None):
        super().__init__(parent)
        self.container_name = container_name
        self.running = True
        self.process = None
        
    def run(self):
        try:
            self.process = subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "100", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            while self.running:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    self.log_update.emit(line)
                    
        except Exception as e:
            self.log_update.emit(f"Monitor Error: {str(e)}\n")
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


class DistributedClientGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client_monitor = None
        self.client_container = None
        # Set by "Test Connection" (or defaults): must match server compose (31883/35672 vs 1883/5672)
        self.remote_mqtt_port = 31883
        self.remote_amqp_port = 35672
        self._dynamic_network_timer = QTimer(self)
        self._dynamic_network_timer.timeout.connect(self._on_dynamic_network_tick)
        self._last_dynamic_base = None
        self.init_ui()

    def _docker_client_entrypoint(self, use_case, protocol, is_unified):
        """
        argv for `docker run ... IMAGE <command>`. The Client Dockerfile CMD defaults to
        FL_Client_MQTT.py; compose files override per protocol. Distributed `docker run`
        must pass the same override or every protocol incorrectly runs MQTT (→ MQTT_PORT e.g. 31883).
        """
        uc = str(use_case).lower().replace(" ", "")
        folder = {
            "emotion": "Emotion_Recognition",
            "mentalstate": "MentalState_Recognition",
            "temperature": "Temperature_Regulation",
        }.get(uc, "Emotion_Recognition")
        if is_unified:
            return ["python3", f"Client/{folder}/FL_Client_Unified.py"]
        script = {
            "mqtt": "FL_Client_MQTT.py",
            "amqp": "FL_Client_AMQP.py",
            "grpc": "FL_Client_gRPC.py",
            "quic": "FL_Client_QUIC.py",
            "http3": "FL_Client_HTTP3.py",
            "dds": "FL_Client_DDS.py",
        }.get(protocol, "FL_Client_MQTT.py")
        return ["python", "-u", f"Client/{folder}/{script}"]

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🌐 Distributed FL Client Dashboard")
        
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        self.setStyleSheet(self.get_stylesheet())
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Configuration tabs
        self.config_tabs = self.create_config_tabs()
        main_layout.addWidget(self.config_tabs)
        
        # Control Buttons
        control_layout = self.create_control_buttons()
        main_layout.addLayout(control_layout)
        
        # Database Management
        db_group = self.create_database_management()
        main_layout.addWidget(db_group)
        
        # Status and Logs
        log_group = self.create_log_section()
        main_layout.addWidget(log_group, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready to connect client")
        
    def create_header(self):
        """Create header section"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #11998e, stop:1 #38ef7d);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout(header)
        title = QLabel("Distributed Federated Learning Client")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Connect a client from a remote PC to the central experiment server")
        subtitle.setStyleSheet("font-size: 13px; color: #f0f0f0;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header

    def create_config_tabs(self):
        """Create tabbed configuration panels"""
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dfe6e9;
                background: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #ecf0f1;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: white;
                font-weight: bold;
            }
        """)
        tabs.addTab(self.create_connection_config(), "Connection")
        tabs.addTab(self.wrap_in_scroll(self.create_client_config()), "Client")
        tabs.addTab(self.wrap_in_scroll(self.create_network_config()), "Network")
        tabs.addTab(self.wrap_in_scroll(self.create_advanced_config()), "Advanced")
        return tabs

    def wrap_in_scroll(self, widget):
        """Wrap a widget in a scroll area for large forms"""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        return scroll
    
    def create_connection_config(self):
        """Create connection configuration section"""
        group = QGroupBox("🔌 Server Connection Settings")
        group.setStyleSheet(self.get_group_style())
        layout = QGridLayout()
        
        # Server IP
        layout.addWidget(QLabel("Server IP Address:"), 0, 0)
        self.server_ip = QLineEdit("192.168.0.102")
        self.server_ip.setPlaceholderText("e.g., 192.168.0.102")
        self.server_ip.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.server_ip, 0, 1)
        
        # Test Connection Button
        self.test_conn_btn = QPushButton("🔍 Test Connection")
        self.test_conn_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.test_conn_btn.clicked.connect(self.test_connection)
        layout.addWidget(self.test_conn_btn, 0, 2)
        
        # Port Configuration (Read-only info)
        info_label = QLabel(
            "Broker ports: Docker-mapped MQTT 31883 / AMQP 35672, or host/macvlan MQTT 1883 / AMQP 5672. "
            "Unified FL also needs TCP gRPC 50051 on the server host (initial model is pulled over gRPC, not MQTT). "
            "Standalone gRPC uses TCP 50051 only (no MQTT). QUIC 4433 | HTTP/3 4434 (UDP) | DDS domain 0. "
            "Test Connection: RL-Unified → MQTT+AMQP TCP, gRPC TCP, QUIC/HTTP3 UDP, DDS (UDP SPDP 7412 + peer fields); "
            "gRPC-only → TCP 50051; MQTT modes → MQTT + gRPC; QUIC/HTTP3 → UDP; DDS-only → ping + SPDP."
        )
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(info_label, 1, 0, 1, 3)

        dds_help = QLabel(
            "DDS static unicast (optional): set all fields if multicast discovery fails across subnets/routers. "
            "Must match the FL server and all client hosts (see config/dds_distributed_unicast.py). "
            "Leave client3 empty if using only 2 clients."
        )
        dds_help.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(dds_help, 2, 0, 1, 3)
        layout.addWidget(QLabel("DDS peer — server host:"), 3, 0)
        self.dds_peer_server = QLineEdit("192.168.0.102")
        self.dds_peer_server.setPlaceholderText("empty = use multicast LAN XML")
        self.dds_peer_server.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.dds_peer_server, 3, 1, 1, 2)
        layout.addWidget(QLabel("DDS peer — client 1 host:"), 4, 0)
        self.dds_peer_client1 = QLineEdit("192.168.0.102")
        self.dds_peer_client1.setPlaceholderText("machine running CLIENT_ID=1")
        self.dds_peer_client1.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.dds_peer_client1, 4, 1, 1, 2)
        layout.addWidget(QLabel("DDS peer — client 2 host:"), 5, 0)
        self.dds_peer_client2 = QLineEdit("192.168.0.100")
        self.dds_peer_client2.setPlaceholderText("machine running CLIENT_ID=2")
        self.dds_peer_client2.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.dds_peer_client2, 5, 1, 1, 2)
        layout.addWidget(QLabel("DDS peer — client 3 host:"), 6, 0)
        self.dds_peer_client3 = QLineEdit("192.168.0.101")
        self.dds_peer_client3.setPlaceholderText("machine running CLIENT_ID=3 (leave empty for 2-client setup)")
        self.dds_peer_client3.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.dds_peer_client3, 6, 1, 1, 2)

        # ── WSL2 / Windows NAT support ──────────────────────────────────────
        wsl2_help = QLabel(
            "WSL2 fix: Docker --network host inside WSL2 binds to the WSL2 virtual IP (172.x.x.x), "
            "not the Windows LAN IP. CycloneDDS then advertises the unreachable WSL2 address to peers. "
            "Enable the checkbox and enter the Windows host LAN IP so DDS advertises a reachable address "
            "(DDS_EXTERNAL_NETWORK_ADDRESS). Also set up UDP port-forwarding — see "
            "scripts/windows_wsl2_dds_portforward.ps1."
        )
        wsl2_help.setWordWrap(True)
        wsl2_help.setStyleSheet(
            "color: #8a6d00; background: #fff8e1; border: 1px solid #ffe082; "
            "font-size: 11px; padding: 6px; border-radius: 3px;"
        )
        layout.addWidget(wsl2_help, 7, 0, 1, 3)

        self.wsl2_mode = QCheckBox("Running inside WSL2 on Windows (DDS NAT fix)")
        self.wsl2_mode.setStyleSheet("font-size: 12px; padding: 4px;")
        self.wsl2_mode.setToolTip(
            "Tick this when the GUI is launched inside WSL2 on Windows.\n"
            "Sets DDS_EXTERNAL_NETWORK_ADDRESS so CycloneDDS advertises the Windows\n"
            "host LAN IP instead of the internal WSL2 virtual IP."
        )
        layout.addWidget(self.wsl2_mode, 8, 0, 1, 3)

        layout.addWidget(QLabel("Windows host LAN IP:"), 9, 0)
        self.wsl2_host_ip = QLineEdit()
        self.wsl2_host_ip.setPlaceholderText("e.g. 192.168.0.100  (same as DDS peer — client N host for this machine)")
        self.wsl2_host_ip.setStyleSheet("padding: 8px; font-size: 12px;")
        self.wsl2_host_ip.setEnabled(False)
        layout.addWidget(self.wsl2_host_ip, 9, 1, 1, 2)

        layout.addWidget(QLabel("WSL2 network interface:"), 10, 0)
        self.wsl2_iface = QLineEdit("eth0")
        self.wsl2_iface.setPlaceholderText("e.g. eth0  (run 'ip link' inside WSL2 to find the interface name)")
        self.wsl2_iface.setStyleSheet("padding: 8px; font-size: 12px;")
        self.wsl2_iface.setEnabled(False)
        layout.addWidget(self.wsl2_iface, 10, 1, 1, 2)

        self.wsl2_mode.toggled.connect(self.wsl2_host_ip.setEnabled)
        self.wsl2_mode.toggled.connect(self.wsl2_iface.setEnabled)
        # Auto-populate Windows host IP from peer field matching this machine's client ID
        self.wsl2_mode.toggled.connect(self._on_wsl2_mode_toggled)

        group.setLayout(layout)
        return group
    
    def create_client_config(self):
        """Create client configuration section"""
        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setSpacing(12)

        group = QGroupBox("⚙️ Client Configuration")
        group.setStyleSheet(self.get_group_style())
        layout = QGridLayout()
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        
        # Client ID / total expected clients
        layout.addWidget(QLabel("Client ID:"), 0, 0)
        self.client_id = QSpinBox()
        self.client_id.setRange(1, 100)
        self.client_id.setValue(3)
        self.client_id.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.client_id, 0, 1)

        layout.addWidget(QLabel("Total Expected Clients:"), 0, 2)
        self.total_clients = QSpinBox()
        self.total_clients.setRange(1, 100)
        self.total_clients.setValue(3)
        self.total_clients.setStyleSheet("padding: 8px; font-size: 12px;")
        self.total_clients.setToolTip("Match the total number of clients expected by the main experiment, including remote clients.")
        layout.addWidget(self.total_clients, 0, 3)
        
        # Use Case
        layout.addWidget(QLabel("Use Case:"), 1, 0)
        self.use_case = QComboBox()
        self.use_case.addItem("emotion", "emotion")
        self.use_case.addItem("mentalstate", "mentalstate")
        self.use_case.addItem("temperature", "temperature")
        self.use_case.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.use_case, 1, 1)
        
        # Protocol Selection Mode
        layout.addWidget(QLabel("Protocol Mode:"), 1, 2)
        self.protocol_mode = QComboBox()
        self.protocol_mode.addItem("RL-Unified (Dynamic Selection)", "rl_unified")
        self.protocol_mode.addItem("MQTT", "mqtt")
        self.protocol_mode.addItem("AMQP", "amqp")
        self.protocol_mode.addItem("gRPC", "grpc")
        self.protocol_mode.addItem("QUIC", "quic")
        self.protocol_mode.addItem("HTTP/3", "http3")
        self.protocol_mode.addItem("DDS", "dds")
        self.protocol_mode.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.protocol_mode, 1, 3)

        self.data_shard_label = QLabel("Data shard (1–3):")
        self.data_shard_combo = QComboBox()
        self.data_shard_combo.addItem("Auto (use Client ID as shard)", None)
        for s in (1, 2, 3):
            self.data_shard_combo.addItem(f"Shard {s}", s)
        self.data_shard_combo.setStyleSheet("padding: 8px; font-size: 12px;")
        self.data_shard_combo.setToolTip(
            "Emotion: Dataset/client_N. Mental state: non-IID partition N (1-based).\n"
            "Auto leaves DATASET_CLIENT_ID unset so the client matches folder/partition to Client ID.\n"
            "Temperature uses one shared CSV — shard has no effect."
        )
        layout.addWidget(self.data_shard_label, 2, 0)
        layout.addWidget(self.data_shard_combo, 2, 1, 1, 3)
        
        # DDS implementation selector (only meaningful when DDS protocol is selected)
        self.dds_impl_label = QLabel("DDS Implementation:")
        self.dds_impl = QComboBox()
        self.dds_impl.addItem("CycloneDDS", "cyclonedds")
        self.dds_impl.addItem("Fast DDS", "fastdds")
        self.dds_impl.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.dds_impl_label, 3, 0)
        layout.addWidget(self.dds_impl, 3, 1)

        # RL-unified mode selector
        layout.addWidget(QLabel("RL-Unified Mode:"), 3, 2)
        rl_mode_layout = QHBoxLayout()
        self.rl_mode_group = QButtonGroup(self)
        self.rl_mode_training = QRadioButton("Training")
        self.rl_mode_inference = QRadioButton("Inference")
        self.rl_mode_inference.setChecked(True)
        self.rl_mode_group.addButton(self.rl_mode_training)
        self.rl_mode_group.addButton(self.rl_mode_inference)
        rl_mode_layout.addWidget(self.rl_mode_training)
        rl_mode_layout.addWidget(self.rl_mode_inference)
        rl_mode_layout.addStretch()
        layout.addLayout(rl_mode_layout, 3, 3)
        
        # GPU Support
        self.gpu_enabled = QCheckBox("Enable GPU (if available)")
        self.gpu_enabled.setStyleSheet("padding: 5px; font-size: 12px;")
        layout.addWidget(self.gpu_enabled, 4, 0, 1, 2)
        
        # Q-learning mode note (mirrors experiment GUI behavior)
        self.ql_convergence_enabled = QCheckBox("End RL training when Q-learning values converge")
        self.ql_convergence_enabled.setStyleSheet("padding: 5px; font-size: 12px;")
        self.ql_convergence_enabled.setToolTip("Automatically enabled in RL training mode, disabled in inference mode.")
        layout.addWidget(self.ql_convergence_enabled, 4, 2, 1, 2)

        self.communication_model_reward_enabled = QCheckBox("Include communication model in RL rewards")
        self.communication_model_reward_enabled.setChecked(True)
        self.communication_model_reward_enabled.setStyleSheet("padding: 5px; font-size: 12px;")
        self.communication_model_reward_enabled.setToolTip(
            "When enabled, RL rewards include the communication-model T_calc penalty. "
            "Disable this to train or run RL without reward influence from the communication model."
        )
        layout.addWidget(self.communication_model_reward_enabled, 5, 0, 1, 4)

        self.mount_shared_data = QCheckBox(
            "Mount project shared_data (metrics JSONL, packet/Q DBs, RL Q-tables)"
        )
        self.mount_shared_data.setChecked(True)
        self.mount_shared_data.setStyleSheet("padding: 5px; font-size: 12px;")
        self.mount_shared_data.setToolTip(
            "Binds <project>/shared_data → /shared_data. Persists per-round client metrics (loss, accuracy, "
            "training/uplink times, battery), packet_logs / q_learning DBs, and RL Q-table .pkl files — same as "
            "experiment Docker compose. Recommended ON for distributed clients so JSONL is not lost in the container."
        )
        layout.addWidget(self.mount_shared_data, 6, 0, 1, 4)

        self.reset_epsilon_on_start = QCheckBox("Reset epsilon to 1.0 (fresh RL exploration)")
        self.reset_epsilon_on_start.setChecked(True)
        self.reset_epsilon_on_start.setStyleSheet("padding: 5px; font-size: 12px;")
        self.reset_epsilon_on_start.setToolTip(
            "Matches the main experiment GUI (Reset Epsilon). Uncheck to resume epsilon/Q progress from saved tables "
            "(equivalent to --no-reset-epsilon on the orchestrator)."
        )
        layout.addWidget(self.reset_epsilon_on_start, 7, 0, 1, 4)
        
        # Termination mode + round cap (must match experiment GUI for fixed-rounds runs)
        layout.addWidget(QLabel("Termination mode:"), 8, 0)
        self.termination_mode_combo = QComboBox()
        self.termination_mode_combo.addItem("End on client convergence (may stop early)", "client_convergence")
        self.termination_mode_combo.addItem("End on fixed rounds (run selected rounds)", "fixed_rounds")
        self.termination_mode_combo.setCurrentIndex(0)  # Preserve existing behavior
        self.termination_mode_combo.setStyleSheet("padding: 5px; font-size: 12px;")
        layout.addWidget(self.termination_mode_combo, 8, 1, 1, 3)
        
        layout.addWidget(QLabel("NUM_ROUNDS:"), 9, 0)
        self.rounds_spinbox = QSpinBox()
        self.rounds_spinbox.setRange(1, 1000)
        self.rounds_spinbox.setValue(10)
        self.rounds_spinbox.setStyleSheet("padding: 5px; font-size: 12px;")
        self.rounds_spinbox.setToolTip("Round cap to keep in sync with the main experiment GUI.")
        layout.addWidget(self.rounds_spinbox, 9, 1)
        
        # Update state when protocol mode changes
        self.protocol_mode.currentIndexChanged.connect(self.update_ql_convergence_visibility)
        self.protocol_mode.currentIndexChanged.connect(self.update_dds_impl_visibility)
        self.use_case.currentIndexChanged.connect(self.update_data_shard_visibility_for_use_case)
        self.rl_mode_training.toggled.connect(self.update_ql_convergence_visibility)
        self.rl_mode_inference.toggled.connect(self.update_ql_convergence_visibility)
        self.update_ql_convergence_visibility()
        self.update_rl_shared_data_visibility()
        self.update_dds_impl_visibility()
        self.update_data_shard_visibility_for_use_case()
        
        group.setLayout(layout)
        outer_layout.addWidget(group)
        outer_layout.addStretch()
        return container
    
    def create_network_config(self):
        """Create network scenario configuration"""
        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setSpacing(12)

        scenario_group = QGroupBox("🌐 Network Conditions (Applied to This Client)")
        scenario_group.setStyleSheet(self.get_group_style())
        scenario_layout = QVBoxLayout()
        
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset Scenario:"))
        self.network_scenario = QComboBox()
        self.network_scenario.addItem("None (No Simulation)", "none")
        self.network_scenario.addItem("Excellent", "excellent")
        self.network_scenario.addItem("Good", "good")
        self.network_scenario.addItem("Moderate", "moderate")
        self.network_scenario.addItem("Poor", "poor")
        self.network_scenario.addItem("Very Poor", "very_poor")
        self.network_scenario.addItem("Satellite", "satellite")
        self.network_scenario.addItem("Light Congestion", "congested_light")
        self.network_scenario.addItem("Moderate Congestion", "congested_moderate")
        self.network_scenario.addItem("Heavy Congestion", "congested_heavy")
        self.network_scenario.addItem(
            "Dynamic (Excellent/Good/Moderate/Poor/Light Congestion, shuffle-bag)", "dynamic"
        )
        self.network_scenario.setStyleSheet("padding: 8px; font-size: 12px;")
        self.network_scenario.currentIndexChanged.connect(self.apply_selected_network_preset)
        preset_row.addWidget(self.network_scenario)
        preset_row.addStretch()
        scenario_layout.addLayout(preset_row)

        self.dynamic_settings_widget = QWidget()
        ds_layout = QHBoxLayout(self.dynamic_settings_widget)
        ds_layout.setContentsMargins(0, 4, 0, 0)
        ds_layout.addWidget(QLabel("Dynamic: re-randomize every"))
        self.dynamic_interval_sec = QSpinBox()
        self.dynamic_interval_sec.setRange(5, 600)
        self.dynamic_interval_sec.setValue(45)
        self.dynamic_interval_sec.setSuffix(" s")
        self.dynamic_interval_sec.setToolTip(
            "While the client runs, tc/netem is redrawn from the dynamic pool "
            "(excellent, good, moderate, poor, congested_light; shuffle-bag by default) on this interval."
        )
        self.dynamic_interval_sec.valueChanged.connect(self._on_dynamic_interval_changed)
        ds_layout.addWidget(self.dynamic_interval_sec)
        self.btn_dynamic_randomize = QPushButton("Randomize scenario now")
        self.btn_dynamic_randomize.setToolTip(
            "Draw the next base scenario (same pool and DYNAMIC_BASE_MODE as run_network_experiments) and apply tc."
        )
        self.btn_dynamic_randomize.clicked.connect(self.on_randomize_dynamic_now)
        self.btn_dynamic_randomize.setEnabled(False)
        ds_layout.addWidget(self.btn_dynamic_randomize)
        ds_layout.addStretch()
        self.dynamic_settings_widget.setVisible(False)
        scenario_layout.addWidget(self.dynamic_settings_widget)

        self.network_preview = QLabel("No network simulation")
        self.network_preview.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 4px;
            padding: 10px;
            font-size: 11px;
            color: #495057;
        """)
        self.network_preview.setWordWrap(True)
        scenario_layout.addWidget(self.network_preview)
        scenario_group.setLayout(scenario_layout)
        outer_layout.addWidget(scenario_group)

        dynamic_group = QGroupBox("🎛️ Dynamic Network Control")
        dynamic_group.setStyleSheet(self.get_group_style())
        dynamic_layout = QGridLayout()

        dynamic_layout.addWidget(QLabel("Latency (ms):"), 0, 0)
        self.latency_slider = QSlider(Qt.Horizontal)
        self.latency_slider.setRange(0, 1000)
        self.latency_slider.setValue(0)
        self.latency_slider.setTickPosition(QSlider.TicksBelow)
        self.latency_slider.setTickInterval(100)
        self.latency_label = QLabel("0 ms")
        self.latency_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.latency_slider.valueChanged.connect(lambda v: self.latency_label.setText(f"{v} ms"))
        self.latency_slider.valueChanged.connect(self.update_network_preview)
        dynamic_layout.addWidget(self.latency_slider, 0, 1)
        dynamic_layout.addWidget(self.latency_label, 0, 2)

        dynamic_layout.addWidget(QLabel("Bandwidth (Mbps):"), 1, 0)
        self.bandwidth_slider = QSlider(Qt.Horizontal)
        self.bandwidth_slider.setRange(1, 1000)
        self.bandwidth_slider.setValue(100)
        self.bandwidth_slider.setTickPosition(QSlider.TicksBelow)
        self.bandwidth_slider.setTickInterval(100)
        self.bandwidth_label = QLabel("100 Mbps")
        self.bandwidth_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.bandwidth_slider.valueChanged.connect(lambda v: self.bandwidth_label.setText(f"{v} Mbps"))
        self.bandwidth_slider.valueChanged.connect(self.update_network_preview)
        dynamic_layout.addWidget(self.bandwidth_slider, 1, 1)
        dynamic_layout.addWidget(self.bandwidth_label, 1, 2)

        dynamic_layout.addWidget(QLabel("Jitter (ms):"), 2, 0)
        self.jitter_slider = QSlider(Qt.Horizontal)
        self.jitter_slider.setRange(0, 100)
        self.jitter_slider.setValue(0)
        self.jitter_slider.setTickPosition(QSlider.TicksBelow)
        self.jitter_slider.setTickInterval(10)
        self.jitter_label = QLabel("0 ms")
        self.jitter_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.jitter_slider.valueChanged.connect(lambda v: self.jitter_label.setText(f"{v} ms"))
        self.jitter_slider.valueChanged.connect(self.update_network_preview)
        dynamic_layout.addWidget(self.jitter_slider, 2, 1)
        dynamic_layout.addWidget(self.jitter_label, 2, 2)

        dynamic_layout.addWidget(QLabel("Packet Loss (%):"), 3, 0)
        self.packet_loss_slider = QSlider(Qt.Horizontal)
        self.packet_loss_slider.setRange(0, 100)
        self.packet_loss_slider.setValue(0)
        self.packet_loss_slider.setTickPosition(QSlider.TicksBelow)
        self.packet_loss_slider.setTickInterval(10)
        self.packet_loss_label = QLabel("0.0 %")
        self.packet_loss_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.packet_loss_slider.valueChanged.connect(lambda v: self.packet_loss_label.setText(f"{v / 10.0:.1f} %"))
        self.packet_loss_slider.valueChanged.connect(self.update_network_preview)
        dynamic_layout.addWidget(self.packet_loss_slider, 3, 1)
        dynamic_layout.addWidget(self.packet_loss_label, 3, 2)

        dynamic_group.setLayout(dynamic_layout)
        outer_layout.addWidget(dynamic_group)

        congestion_group = QGroupBox("🚦 Traffic Congestion")
        congestion_group.setStyleSheet(self.get_group_style())
        congestion_layout = QHBoxLayout()
        self.enable_congestion = QCheckBox("Enable Traffic Generator-Based Congestion")
        self.enable_congestion.setStyleSheet("font-size: 13px; padding: 5px;")
        self.enable_congestion.toggled.connect(self.update_network_preview)
        congestion_layout.addWidget(self.enable_congestion)
        congestion_layout.addWidget(QLabel("Congestion Level:"))
        self.congestion_level = QComboBox()
        self.congestion_level.addItems(["Light", "Moderate", "Heavy"])
        self.congestion_level.setCurrentText("Moderate")
        self.congestion_level.setStyleSheet("padding: 5px;")
        self.congestion_level.currentIndexChanged.connect(self.update_network_preview)
        congestion_layout.addWidget(self.congestion_level)
        congestion_layout.addStretch()
        congestion_group.setLayout(congestion_layout)
        outer_layout.addWidget(congestion_group)

        outer_layout.addStretch()
        self.apply_selected_network_preset()
        return container

    def create_advanced_config(self):
        """Create advanced runtime options to match experiment GUI"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        quant_group = QGroupBox("🔢 Model Quantization")
        quant_group.setStyleSheet(self.get_group_style())
        quant_layout = QVBoxLayout()
        self.quantization_enabled = QCheckBox("Enable Quantization")
        self.quantization_enabled.setStyleSheet("font-size: 13px; padding: 5px;")
        self.quantization_enabled.toggled.connect(self.toggle_quantization_options)
        quant_layout.addWidget(self.quantization_enabled)

        quant_options = QWidget()
        quant_options_layout = QGridLayout()
        quant_options_layout.addWidget(QLabel("Quantization Bits:"), 0, 0)
        self.quant_bits = QComboBox()
        self.quant_bits.addItems(["4", "8", "16", "32"])
        self.quant_bits.setCurrentText("8")
        self.quant_bits.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_bits, 0, 1)
        quant_options_layout.addWidget(QLabel("Strategy:"), 0, 2)
        self.quant_strategy = QComboBox()
        self.quant_strategy.addItems(["full_quantization", "parameter_quantization", "activation_quantization"])
        self.quant_strategy.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_strategy, 0, 3)
        self.quant_symmetric = QCheckBox("Use Symmetric Quantization")
        self.quant_symmetric.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_symmetric, 1, 0, 1, 2)
        self.quant_per_channel = QCheckBox("Per-Channel Quantization")
        self.quant_per_channel.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_per_channel, 1, 2, 1, 2)
        quant_options.setLayout(quant_options_layout)
        quant_options.setEnabled(False)
        self.quant_options_widget = quant_options
        quant_layout.addWidget(quant_options)
        quant_group.setLayout(quant_layout)
        layout.addWidget(quant_group)

        compression_group = QGroupBox("📦 Model Compression")
        compression_group.setStyleSheet(self.get_group_style())
        compression_layout = QVBoxLayout()
        self.compression_enabled = QCheckBox("Enable Compression")
        self.compression_enabled.setStyleSheet("font-size: 13px; padding: 5px;")
        self.compression_enabled.toggled.connect(self.toggle_compression_options)
        compression_layout.addWidget(self.compression_enabled)

        comp_options = QWidget()
        comp_options_layout = QHBoxLayout()
        comp_options_layout.addWidget(QLabel("Algorithm:"))
        self.compression_algo = QComboBox()
        self.compression_algo.addItems(["gzip", "lz4", "zstd", "snappy"])
        self.compression_algo.setCurrentText("gzip")
        self.compression_algo.setStyleSheet("padding: 5px;")
        comp_options_layout.addWidget(self.compression_algo)
        comp_options_layout.addWidget(QLabel("Level:"))
        self.compression_level = QSpinBox()
        self.compression_level.setRange(1, 9)
        self.compression_level.setValue(6)
        self.compression_level.setStyleSheet("padding: 5px;")
        comp_options_layout.addWidget(self.compression_level)
        comp_options_layout.addStretch()
        comp_options.setLayout(comp_options_layout)
        comp_options.setEnabled(False)
        self.comp_options_widget = comp_options
        compression_layout.addWidget(comp_options)
        compression_group.setLayout(compression_layout)
        layout.addWidget(compression_group)

        pruning_group = QGroupBox("✂️ Model Pruning")
        pruning_group.setStyleSheet(self.get_group_style())
        pruning_layout = QVBoxLayout()
        self.pruning_enabled = QCheckBox("Enable Pruning")
        self.pruning_enabled.setStyleSheet("font-size: 13px; padding: 5px;")
        self.pruning_enabled.toggled.connect(self.toggle_pruning_options)
        pruning_layout.addWidget(self.pruning_enabled)

        pruning_options = QWidget()
        pruning_options_layout = QHBoxLayout()
        pruning_options_layout.addWidget(QLabel("Pruning Ratio:"))
        self.pruning_ratio = QSlider(Qt.Horizontal)
        self.pruning_ratio.setRange(0, 90)
        self.pruning_ratio.setValue(50)
        self.pruning_ratio.setTickPosition(QSlider.TicksBelow)
        self.pruning_ratio.setTickInterval(10)
        self.pruning_ratio_label = QLabel("50%")
        self.pruning_ratio_label.setStyleSheet("font-weight: bold; min-width: 50px;")
        self.pruning_ratio.valueChanged.connect(lambda v: self.pruning_ratio_label.setText(f"{v}%"))
        pruning_options_layout.addWidget(self.pruning_ratio)
        pruning_options_layout.addWidget(self.pruning_ratio_label)
        pruning_options.setLayout(pruning_options_layout)
        pruning_options.setEnabled(False)
        self.pruning_options_widget = pruning_options
        pruning_layout.addWidget(pruning_options)
        pruning_group.setLayout(pruning_layout)
        layout.addWidget(pruning_group)

        other_group = QGroupBox("⚙️ Other Options")
        other_group.setStyleSheet(self.get_group_style())
        other_layout = QGridLayout()
        self.save_checkpoints = QCheckBox("Save Model Checkpoints")
        self.save_checkpoints.setChecked(True)
        self.save_checkpoints.setStyleSheet("padding: 5px;")
        other_layout.addWidget(self.save_checkpoints, 0, 0)
        self.verbose_logging = QCheckBox("Verbose Logging")
        self.verbose_logging.setStyleSheet("padding: 5px;")
        other_layout.addWidget(self.verbose_logging, 0, 1)
        self.enable_tensorboard = QCheckBox("Enable TensorBoard")
        self.enable_tensorboard.setStyleSheet("padding: 5px;")
        other_layout.addWidget(self.enable_tensorboard, 1, 0)
        self.profile_performance = QCheckBox("Profile Performance")
        self.profile_performance.setStyleSheet("padding: 5px;")
        other_layout.addWidget(self.profile_performance, 1, 1)
        other_group.setLayout(other_layout)
        layout.addWidget(other_group)

        layout.addStretch()
        return container
    
    def create_control_buttons(self):
        """Create control buttons"""
        layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶️ Start Client")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 30px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:pressed { background-color: #1e7e34; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.start_btn.clicked.connect(self.start_client)
        
        self.stop_btn = QPushButton("⏹️ Stop Client")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 30px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:pressed { background-color: #bd2130; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.stop_btn.clicked.connect(self.stop_client)
        self.stop_btn.setEnabled(False)
        
        self.rebuild_btn = QPushButton("🔨 Rebuild Image")
        self.rebuild_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 30px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:pressed { background-color: #004085; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.rebuild_btn.clicked.connect(self.rebuild_client_image)
        
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.rebuild_btn)
        layout.addStretch()
        
        return layout
    
    def create_database_management(self):
        """Create database management section"""
        group = QGroupBox("🗄️ Database Management")
        group.setStyleSheet(self.get_group_style())
        layout = QHBoxLayout()
        
        self.reset_db_btn = QPushButton("🗑️ Reset Databases")
        self.reset_db_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #ee5a6f; }
            QPushButton:pressed { background-color: #dc4a5f; }
        """)
        self.reset_db_btn.clicked.connect(self.reset_databases)
        self.reset_db_btn.setToolTip("Delete all packet_logs and q_learning database files from shared_data directory")
        
        info_label = QLabel("This will delete all existing database files (packet_logs_*.db and q_learning_*.db)")
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        
        layout.addWidget(self.reset_db_btn)
        layout.addWidget(info_label)
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def update_ql_convergence_visibility(self):
        """Update Q-learning convergence checkbox visibility based on protocol mode"""
        is_rl_unified = (self.protocol_mode.currentData() == "rl_unified")
        training_selected = is_rl_unified and self.rl_mode_training.isChecked()
        self.rl_mode_training.setEnabled(is_rl_unified)
        self.rl_mode_inference.setEnabled(is_rl_unified)
        self.ql_convergence_enabled.setChecked(training_selected)
        self.ql_convergence_enabled.setEnabled(False)
        self.communication_model_reward_enabled.setEnabled(is_rl_unified)
        self.update_rl_shared_data_visibility()

    def update_rl_shared_data_visibility(self):
        """Always show shared_data mount (metrics + DBs + RL tables). Epsilon reset only for RL-unified training."""
        if not hasattr(self, "mount_shared_data"):
            return
        is_rl_unified = self.protocol_mode.currentData() == "rl_unified"
        training_selected = is_rl_unified and self.rl_mode_training.isChecked()
        self.mount_shared_data.setVisible(True)
        self.reset_epsilon_on_start.setVisible(training_selected)

    def _sync_reset_epsilon_flag_for_distributed(self, shared_data_path):
        """If resuming epsilon, remove reset_epsilon_flag.txt so a copied shared_data tree does not force reset."""
        if not self.is_rl_training_mode() or self.reset_epsilon_on_start.isChecked():
            return
        flag_path = os.path.join(shared_data_path, "reset_epsilon_flag.txt")
        try:
            if os.path.isfile(flag_path):
                os.remove(flag_path)
                self.log_text.append(
                    "Removed reset_epsilon_flag.txt (resume epsilon; matches orchestrator --no-reset-epsilon).\n"
                )
        except OSError as e:
            self.log_text.append(f"⚠️ Could not remove reset_epsilon_flag.txt: {e}\n")

    def _write_current_rl_network_scenario_file(self, shared_data_path: str, scenario_label: str) -> None:
        """
        Hint for unified FL clients: fine-grained scenario label (especially ``dynamic`` draws).
        Read by ``FL_Client_Unified`` as ``/shared_data/current_rl_network_scenario.txt``.
        """
        try:
            os.makedirs(shared_data_path, exist_ok=True)
            path = os.path.join(shared_data_path, "current_rl_network_scenario.txt")
            sl = (scenario_label or "default").strip().lower() or "default"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"scenario={sl}\n")
        except OSError as e:
            self.log_text.append(f"⚠️ Could not write current_rl_network_scenario.txt: {e}\n")

    def _gui_resolved_network_scenario_label(self, network_scenario_combo_data) -> str:
        """Map GUI combo value to the RL / SQLite network-scenario string (dynamic → current draw)."""
        data = network_scenario_combo_data
        if data in (None, "none"):
            return "none"
        if data == "dynamic":
            if not getattr(self, "_last_dynamic_base", None):
                self.apply_random_dynamic_base()
            return str(self._last_dynamic_base or "moderate").strip().lower()
        return str(data).strip().lower() or "default"

    def update_dds_impl_visibility(self):
        """Enable DDS implementation selector only when DDS protocol is selected"""
        is_dds = (self.protocol_mode.currentData() == "dds")
        self.dds_impl_label.setEnabled(is_dds)
        self.dds_impl.setEnabled(is_dds)

    def update_data_shard_visibility_for_use_case(self):
        """Enable data-shard selector for emotion and mental state only."""
        if not hasattr(self, "data_shard_combo"):
            return
        uc = self.use_case.currentData()
        en = uc in ("emotion", "mentalstate")
        self.data_shard_combo.setEnabled(en)
        self.data_shard_label.setEnabled(en)

    def is_rl_training_mode(self):
        """Return True when RL-unified is configured in training mode"""
        return self.protocol_mode.currentData() == "rl_unified" and self.rl_mode_training.isChecked()

    def toggle_quantization_options(self, enabled):
        """Toggle quantization options"""
        self.quant_options_widget.setEnabled(enabled)

    def toggle_compression_options(self, enabled):
        """Toggle compression options"""
        self.comp_options_widget.setEnabled(enabled)

    def toggle_pruning_options(self, enabled):
        """Toggle pruning options"""
        self.pruning_options_widget.setEnabled(enabled)
    
    def create_log_section(self):
        """Create log and status section"""
        group = QGroupBox("📊 Client Status and Logs")
        group.setStyleSheet(self.get_group_style())
        layout = QVBoxLayout()
        
        # Status info
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Not Running")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #6c757d;")
        status_layout.addWidget(self.status_label)
        
        self.connection_status = QLabel("● Disconnected")
        self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
        status_layout.addWidget(self.connection_status)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.log_text)
        
        group.setLayout(layout)
        return group
    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                font-size: 12px;
                color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #34495e;
            }
        """
    
    def get_group_style(self):
        """Get group box style"""
        return """
            QGroupBox {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
        """
    
    def update_network_preview(self):
        """Update network scenario preview"""
        profile = self.get_effective_network_profile()
        data = self.network_scenario.currentData()
        if data == "dynamic" and self._last_dynamic_base:
            scenario_name = (
                f"Dynamic — current draw: {self._last_dynamic_base}"
            )
        else:
            scenario_name = self.network_scenario.currentText()
        preview = (
            f"📊 {scenario_name}\n"
            f"Latency: {profile['latency']} ms | "
            f"Bandwidth: {profile['bandwidth']} Mbps | "
            f"Jitter: {profile['jitter']} ms | "
            f"Loss: {profile['packet_loss']:.1f}%"
        )
        if self.enable_congestion.isChecked():
            preview += f"\nCongestion: {self.congestion_level.currentText()}"
        self.network_preview.setText(preview)

    def apply_random_dynamic_base(self):
        """Draw next dynamic base via network_simulator (shuffle-bag default; matches run_network_experiments)."""
        _ns_dir = os.path.dirname(os.path.abspath(__file__))
        if _ns_dir not in sys.path:
            sys.path.insert(0, _ns_dir)
        from network_simulator import draw_dynamic_base_scenario

        base = draw_dynamic_base_scenario()
        self._last_dynamic_base = base
        preset = NETWORK_SCENARIO_PRESETS[base]
        for slider in (self.latency_slider, self.bandwidth_slider, self.jitter_slider, self.packet_loss_slider):
            slider.blockSignals(True)
        self.latency_slider.setValue(int(preset["latency"]))
        self.bandwidth_slider.setValue(int(preset["bandwidth"]))
        self.jitter_slider.setValue(int(preset["jitter"]))
        self.packet_loss_slider.setValue(int(round(preset["packet_loss"] * 10)))
        for slider in (self.latency_slider, self.bandwidth_slider, self.jitter_slider, self.packet_loss_slider):
            slider.blockSignals(False)
        self.latency_label.setText(f"{self.latency_slider.value()} ms")
        self.bandwidth_label.setText(f"{self.bandwidth_slider.value()} Mbps")
        self.jitter_label.setText(f"{self.jitter_slider.value()} ms")
        self.packet_loss_label.setText(f"{self.packet_loss_slider.value() / 10.0:.1f} %")
        self.update_network_preview()

    def apply_selected_network_preset(self):
        """Apply the selected preset values to the manual network controls"""
        data = self.network_scenario.currentData()
        if data == "dynamic":
            _ns_dir = os.path.dirname(os.path.abspath(__file__))
            if _ns_dir not in sys.path:
                sys.path.insert(0, _ns_dir)
            from network_simulator import reset_dynamic_base_scenario_draw

            reset_dynamic_base_scenario_draw()
            self.apply_random_dynamic_base()
            self.dynamic_settings_widget.setVisible(True)
            return
        self._last_dynamic_base = None
        self.dynamic_settings_widget.setVisible(False)
        preset = NETWORK_SCENARIO_PRESETS.get(data, NETWORK_SCENARIO_PRESETS["none"])
        for slider in (self.latency_slider, self.bandwidth_slider, self.jitter_slider, self.packet_loss_slider):
            slider.blockSignals(True)
        self.latency_slider.setValue(int(preset["latency"]))
        self.bandwidth_slider.setValue(int(preset["bandwidth"]))
        self.jitter_slider.setValue(int(preset["jitter"]))
        self.packet_loss_slider.setValue(int(round(preset["packet_loss"] * 10)))
        for slider in (self.latency_slider, self.bandwidth_slider, self.jitter_slider, self.packet_loss_slider):
            slider.blockSignals(False)
        self.latency_label.setText(f"{self.latency_slider.value()} ms")
        self.bandwidth_label.setText(f"{self.bandwidth_slider.value()} Mbps")
        self.jitter_label.setText(f"{self.jitter_slider.value()} ms")
        self.packet_loss_label.setText(f"{self.packet_loss_slider.value() / 10.0:.1f} %")
        self.update_network_preview()

    def get_effective_network_profile(self):
        """Build the final tc/netem profile from UI controls"""
        profile = {
            "latency": self.latency_slider.value(),
            "bandwidth": self.bandwidth_slider.value(),
            "jitter": self.jitter_slider.value(),
            "packet_loss": self.packet_loss_slider.value() / 10.0,
        }
        if self.enable_congestion.isChecked():
            congestion = CONGESTION_PRESETS[self.congestion_level.currentText().lower()]
            profile["latency"] += congestion["latency"]
            profile["jitter"] += congestion["jitter"]
            profile["packet_loss"] += congestion["packet_loss"]
            profile["bandwidth"] = max(1, int(profile["bandwidth"] * congestion["bandwidth_factor"]))
        return profile
    
    def _ping_server_host(self, server_ip: str):
        """Return True if ICMP ping succeeds, False if it fails, None if ping is unavailable."""
        if not shutil.which("ping"):
            return None
        try:
            if sys.platform == "win32":
                cmd = ["ping", "-n", "1", "-w", "3000", server_ip]
            elif sys.platform == "darwin":
                # macOS: -W is milliseconds per reply
                cmd = ["ping", "-c", "1", "-W", "2000", server_ip]
            else:
                # Linux (iputils): -W is seconds
                cmd = ["ping", "-c", "1", "-W", "3", server_ip]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            return False

    def _test_connection_dds_only(self, server_ip: str):
        """DDS-only experiments do not run MQTT; verify host reachability (ping) and remind about UDP."""
        self.log_text.append(
            f"\n🔍 DDS-only: skipping MQTT (broker not required). "
            f"Checking {server_ip} — SPDP UDP + ICMP ping...\n"
        )
        spdp_srv_port = self._cyclone_spdp_port_domain0(1)
        dds_spdp_ok = self._udp_port_probe(server_ip, spdp_srv_port)
        self.log_text.append(
            f"{'✅' if dds_spdp_ok else 'ℹ️'} DDS SPDP UDP {spdp_srv_port} on {server_ip} "
            f"({'reply — Cyclone may be listening' if dds_spdp_ok else 'no reply — normal if server DDS not running; allow UDP ~7400–7500'})\n"
        )
        self.statusBar().showMessage("Testing host reachability (DDS)...")
        ok = self._ping_server_host(server_ip)
        if ok is True:
            self.log_text.append(
                f"✅ Host {server_ip} answered ping. DDS uses UDP multicast/7400–7500; "
                f"ensure the same CYCLONEDDS_URI and firewall rules as in the docs.\n"
            )
            self.connection_status.setText("● Reachable (ping)")
            self.connection_status.setStyleSheet("font-size: 12px; color: #28a745;")
            QMessageBox.information(
                self,
                "Host reachable (DDS)",
                f"Ping to {server_ip} succeeded.\n"
                f"SPDP UDP {spdp_srv_port}: {'open (best-effort)' if dds_spdp_ok else 'no nc reply (optional)'}\n\n"
                "DDS does not use MQTT; no broker is required for this mode.\n"
                "Discovery uses UDP (multicast on the LAN or static DDS_PEER_*); allow UDP 7400–7500 between hosts if needed.",
            )
            self.statusBar().showMessage("DDS: host reachable")
            return
        if ok is False:
            self.log_text.append(
                f"❌ Ping to {server_ip} failed (ICMP blocked or host down).\n"
                f"   You can still start the client if the IP is correct and UDP 7400–7500 is allowed.\n"
            )
            self.connection_status.setText("● Ping failed")
            self.connection_status.setStyleSheet("font-size: 12px; color: #e67e22;")
            QMessageBox.warning(
                self,
                "Ping failed (DDS)",
                f"No ICMP reply from {server_ip}.\n\n"
                "Many networks block ping; DDS may still work if:\n"
                "• The server IP is correct\n"
                "• UDP 7400–7500 is allowed (firewall)\n"
                "• Multicast is allowed on the LAN (see cyclonedds-multicast-lan.xml)\n\n"
                "You can start the client anyway.",
            )
            self.statusBar().showMessage("DDS: ping failed — check IP/firewall")
            return
        # ok is None — ping not installed
        self.log_text.append(
            "⚠️ `ping` not found; cannot verify host. Start the client if the server IP is correct.\n"
        )
        self.connection_status.setText("● Not verified")
        self.connection_status.setStyleSheet("font-size: 12px; color: #e67e22;")
        QMessageBox.information(
            self,
            "DDS — no ping",
            "The `ping` command was not found; host reachability was not tested.\n\n"
            "DDS does not require MQTT. Ensure the server IP is correct and UDP 7400–7500 is allowed.",
        )
        self.statusBar().showMessage("DDS: install ping to test reachability")

    def _udp_port_probe(self, host: str, port: int) -> bool:
        """Best-effort UDP reachability (QUIC/HTTP3). May be inconclusive if ICMP/nc differ."""
        try:
            r = subprocess.run(
                ["timeout", "6", "nc", "-u", "-z", "-v", "-w", "3", host, str(port)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            return False

    @staticmethod
    def _cyclone_spdp_port_domain0(participant_index: int) -> int:
        """CycloneDDS domain 0 SPDP port: 7410 + 2 * ParticipantIndex (config/dds_distributed_unicast.py)."""
        return 7410 + 2 * int(participant_index)

    def _test_connection_quic_http3(self, server_ip: str, protocol: str):
        """QUIC and HTTP/3 use UDP; do not test MQTT."""
        port = 4433 if protocol == "quic" else 4434
        name = "QUIC" if protocol == "quic" else "HTTP/3"
        self.log_text.append(
            f"\n🔍 {name}: testing UDP/{port} to {server_ip} (not MQTT)...\n"
        )
        self.statusBar().showMessage(f"Testing {name} UDP {port}...")
        ping_ok = self._ping_server_host(server_ip)
        if ping_ok:
            self.log_text.append(f"✅ ICMP ping to {server_ip} succeeded.\n")
        else:
            self.log_text.append(
                f"⚠️ ICMP ping to {server_ip} failed or unavailable (UDP may still work).\n"
            )

        udp_ok = self._udp_port_probe(server_ip, port)
        if udp_ok:
            self.log_text.append(
                f"✅ UDP port {port} probe succeeded (`nc -u -z`). {name} is likely reachable.\n"
            )
            self.connection_status.setText("● Reachable (UDP)")
            self.connection_status.setStyleSheet("font-size: 12px; color: #28a745;")
            QMessageBox.information(
                self,
                f"{name} — UDP OK",
                f"Host {server_ip}: UDP port {port} responded to a probe.\n\n"
                f"{name} uses QUIC over UDP (not TCP/MQTT). Ensure the server publishes "
                f"\"{port}:{port}/udp\" in Docker and that firewalls allow UDP/{port}.",
            )
            self.statusBar().showMessage(f"{name}: UDP {port} OK")
            return

        self.log_text.append(
            f"❌ UDP probe to {server_ip}:{port} did not succeed.\n"
            f"   Common causes: Docker mapped TCP only (use \"{port}:{port}/udp\"), "
            f"or host firewall blocking UDP/{port}.\n"
        )
        self.connection_status.setText("● UDP blocked / inconclusive")
        self.connection_status.setStyleSheet("font-size: 12px; color: #e67e22;")
        QMessageBox.warning(
            self,
            f"{name} — UDP not verified",
            f"Could not confirm UDP/{port} to {server_ip} (QUIC/HTTP3).\n\n"
            f"• Docker: map UDP explicitly, e.g. \"{port}:{port}/udp\" for the HTTP/3 server.\n"
            f"• Firewall: allow UDP/{port} from this machine to the server.\n"
            f"• `nc` may be missing; install netcat-openbsd.\n\n"
            f"This mode does not use MQTT — ignore MQTT broker checks.",
        )
        self.statusBar().showMessage(f"{name}: fix UDP {port} mapping / firewall")

    def _tcp_port_probe(self, host: str, port: int) -> bool:
        """Best-effort TCP reachability (gRPC, broker checks)."""
        try:
            r = subprocess.run(
                ["timeout", "6", "nc", "-z", "-v", "-w", "3", host, str(port)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            return False

    def _test_connection_grpc_only(self, server_ip: str):
        """Standalone gRPC uses HTTP/2 over TCP 50051; no MQTT broker."""
        grpc_port = 50051
        self.log_text.append(
            f"\n🔍 gRPC-only: testing TCP/{grpc_port} to {server_ip} (not MQTT)...\n"
        )
        self.statusBar().showMessage(f"Testing gRPC TCP {grpc_port}...")
        ping_ok = self._ping_server_host(server_ip)
        if ping_ok:
            self.log_text.append(f"✅ ICMP ping to {server_ip} succeeded.\n")
        else:
            self.log_text.append(
                f"⚠️ ICMP ping to {server_ip} failed or unavailable (gRPC may still work).\n"
            )

        tcp_ok = self._tcp_port_probe(server_ip, grpc_port)
        if tcp_ok:
            self.log_text.append(
                f"✅ TCP port {grpc_port} open (`nc -z`). gRPC server is reachable from this PC.\n"
            )
            self.connection_status.setText("● Reachable (gRPC)")
            self.connection_status.setStyleSheet("font-size: 12px; color: #28a745;")
            QMessageBox.information(
                self,
                "gRPC — TCP OK",
                f"Host {server_ip}: TCP port {grpc_port} is reachable.\n\n"
                f"Standalone gRPC does not use MQTT. Ensure the FL gRPC server container is up and "
                f"Docker publishes \"{grpc_port}:{grpc_port}\" (TCP) on the host.",
            )
            self.statusBar().showMessage(f"gRPC: TCP {grpc_port} OK")
            return

        self.log_text.append(
            f"❌ TCP {grpc_port} to {server_ip} not reachable from this PC.\n"
            f"   Open {grpc_port}/tcp on the server firewall (e.g. sudo ufw allow {grpc_port}/tcp) and "
            f"ensure Docker maps host port {grpc_port} to the gRPC server.\n"
        )
        self.connection_status.setText("● gRPC blocked")
        self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
        QMessageBox.warning(
            self,
            "gRPC — TCP failed",
            f"Could not connect to TCP/{grpc_port} on {server_ip}.\n\n"
            f"• Firewall on the server: allow {grpc_port}/tcp (e.g. ufw allow {grpc_port}/tcp).\n"
            f"• Docker: fl-server-grpc-* must publish \"{grpc_port}:{grpc_port}\" or \"{grpc_port}:{grpc_port}/tcp\".\n"
            f"• NAT/router: forward TCP {grpc_port} to the server host if needed.\n\n"
            f"This mode does not use MQTT — ignore MQTT broker checks.",
        )
        self.statusBar().showMessage(f"gRPC: open TCP {grpc_port} / firewall")

    def _test_connection_unified(self, server_ip: str):
        """RL-Unified needs MQTT, AMQP, gRPC (TCP), QUIC & HTTP/3 (UDP). DDS is separate (UDP multicast/unicast)."""
        self.log_text.append(
            f"\n🔍 RL-Unified: checking {server_ip} — MQTT, AMQP, gRPC, QUIC, HTTP/3, "
            f"DDS (UDP SPDP / peers; not a TCP port like the others)...\n"
        )
        self.statusBar().showMessage("Testing RL-Unified endpoints…")

        ping_ok = self._ping_server_host(server_ip)
        if ping_ok is True:
            self.log_text.append(f"✅ ICMP ping to {server_ip} succeeded.\n")
        elif ping_ok is False:
            self.log_text.append(
                f"⚠️ ICMP ping to {server_ip} failed or blocked (brokers may still be reachable).\n"
            )
        else:
            self.log_text.append("⚠️ `ping` not installed; skipping host ICMP check.\n")

        candidates = [
            (31883, 35672, "Docker-mapped (e.g. docker-compose-unified-emotion.bridge.yml)"),
            (1883, 5672, "Standard broker ports (e.g. macvlan / host / config compose)"),
        ]

        mqtt_p = amqp_p = None
        desc = ""
        for mp, ap, d in candidates:
            if self._tcp_port_probe(server_ip, mp):
                mqtt_p, amqp_p, desc = mp, ap, d
                self.remote_mqtt_port = mp
                self.remote_amqp_port = ap
                break

        if mqtt_p is None:
            self.log_text.append(
                f"❌ MQTT not reachable on {server_ip}:31883 or :1883.\n"
                f"   Start brokers + unified stack; open firewall TCP 31883/1883 from this PC.\n"
            )
            self.connection_status.setText("● Disconnected")
            self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
            QMessageBox.warning(
                self,
                "RL-Unified — MQTT",
                f"Cannot reach MQTT on {server_ip} (tried TCP 31883 and 1883).\n\n"
                f"Remote RL-Unified clients need the MQTT broker published on the host.\n"
                f"Check server IP, docker compose port maps, and firewall.",
            )
            self.statusBar().showMessage("RL-Unified: MQTT failed")
            return

        self.log_text.append(f"✅ MQTT TCP {mqtt_p} — {desc}\n")
        mqtt_ok = True

        amqp_ok = self._tcp_port_probe(server_ip, amqp_p)
        self.log_text.append(
            f"{'✅' if amqp_ok else '❌'} AMQP TCP {amqp_p} ({'RabbitMQ' if amqp_ok else 'not reachable from this PC'})\n"
        )

        grpc_port = 50051
        grpc_ok = self._tcp_port_probe(server_ip, grpc_port)
        self.log_text.append(
            f"{'✅' if grpc_ok else '❌'} gRPC TCP {grpc_port} "
            f"({'unified server / initial model' if grpc_ok else 'required for initial global model'})\n"
        )

        quic_ok = self._udp_port_probe(server_ip, 4433)
        http3_ok = self._udp_port_probe(server_ip, 4434)
        self.log_text.append(
            f"{'✅' if quic_ok else '❌'} QUIC UDP 4433 ({'OK' if quic_ok else 'allow UDP / Docker 4433:4433/udp'})\n"
        )
        self.log_text.append(
            f"{'✅' if http3_ok else '❌'} HTTP/3 UDP 4434 ({'OK' if http3_ok else 'allow UDP / Docker 4434:4434/udp'})\n"
        )

        # DDS: not TCP — probe optional SPDP UDP on server (participant index 1 → port 7412) + UI peer fields
        spdp_srv_port = self._cyclone_spdp_port_domain0(1)
        dds_spdp_ok = self._udp_port_probe(server_ip, spdp_srv_port)
        ps = getattr(self, "dds_peer_server", None)
        p1 = getattr(self, "dds_peer_client1", None)
        p2 = getattr(self, "dds_peer_client2", None)
        peers_filled = bool(
            ps and p1 and p2
            and ps.text().strip() and p1.text().strip() and p2.text().strip()
        )
        self.log_text.append(
            f"{'✅' if dds_spdp_ok else 'ℹ️'} DDS SPDP UDP {spdp_srv_port} on {server_ip} "
            f"({'reply — Cyclone may be listening on server' if dds_spdp_ok else 'no reply — normal if unified server has no DDS participant yet; allow UDP ~7400–7500 + SPDP'})\n"
        )
        if peers_filled:
            self.log_text.append(
                "✅ DDS_PEER_*: peer fields set (static unicast; must match the experiment server and all client hosts).\n"
            )
        else:
            self.log_text.append(
                "ℹ️ DDS: set all three DDS peer fields for cross-host static unicast, or rely on multicast LAN XML.\n"
            )

        transport_ok = mqtt_ok and amqp_ok and grpc_ok and quic_ok and http3_ok
        core_ok = mqtt_ok and grpc_ok

        if transport_ok:
            self.connection_status.setText("● Reachable (full)")
            self.connection_status.setStyleSheet("font-size: 12px; color: #28a745;")
            QMessageBox.information(
                self,
                "RL-Unified — all probes OK",
                f"Server {server_ip}:\n"
                f"• MQTT TCP {mqtt_p} ✓\n"
                f"• AMQP TCP {amqp_p} ✓\n"
                f"• gRPC TCP {grpc_port} ✓\n"
                f"• QUIC UDP 4433 ✓\n"
                f"• HTTP/3 UDP 4434 ✓\n"
                f"• DDS: UDP discovery (not TCP); SPDP probe {spdp_srv_port} "
                f"{'✓' if dds_spdp_ok else '— (optional)'}, peers in UI "
                f"{'✓' if peers_filled else '— fill for static unicast'}\n\n"
                f"DDS cannot use the same check as MQTT/gRPC; match DDS_PEER_* / CYCLONEDDS_URI with the server.",
            )
            self.statusBar().showMessage("RL-Unified: all endpoints OK")
            return

        if not core_ok:
            self.connection_status.setText("● Blocked (MQTT/gRPC)")
            self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
            fail = []
            if not grpc_ok:
                fail.append(f"gRPC TCP {grpc_port} (required for initial model)")
            QMessageBox.warning(
                self,
                "RL-Unified — critical failure",
                f"MQTT or gRPC is not reachable on {server_ip}.\n\n"
                f"Unified clients need MQTT for control and gRPC TCP {grpc_port} for the first global model.\n\n"
                + ("\n".join(fail) if fail else "Open firewall and confirm Docker publishes these ports."),
            )
            self.statusBar().showMessage("RL-Unified: fix MQTT/gRPC")
            return

        # Core OK but some optional transports missing
        self.connection_status.setText("● Partial")
        self.connection_status.setStyleSheet("font-size: 12px; color: #e67e22;")
        parts = []
        if not amqp_ok:
            parts.append(f"AMQP TCP {amqp_p} — RL cannot use AMQP until open.")
        if not quic_ok:
            parts.append("QUIC UDP 4433 — map 4433:4433/udp, firewall UDP/4433.")
        if not http3_ok:
            parts.append("HTTP/3 UDP 4434 — map 4434:4434/udp, firewall UDP/4434.")
        msg = (
            f"Server {server_ip}: MQTT and gRPC are OK; some other transports failed:\n\n"
            + "\n".join(parts)
            + "\n\nRL can still run, but protocol selection will skip unreachable transports."
        )
        QMessageBox.warning(self, "RL-Unified — partial", msg)
        self.statusBar().showMessage("RL-Unified: partial — open AMQP/QUIC/HTTP3 if needed")

    def test_connection(self):
        """Test connection to server (try Docker-mapped ports first, then standard broker ports)."""
        server_ip = self.server_ip.text().strip()
        
        if not server_ip:
            QMessageBox.warning(self, "Error", "Please enter server IP address!")
            return

        protocol = self.protocol_mode.currentData() or self.protocol_mode.currentText()
        if protocol == "dds":
            self._test_connection_dds_only(server_ip)
            return
        if protocol == "http3":
            self._test_connection_quic_http3(server_ip, "http3")
            return
        if protocol == "quic":
            self._test_connection_quic_http3(server_ip, "quic")
            return
        if protocol == "grpc":
            self._test_connection_grpc_only(server_ip)
            return
        if protocol == "rl_unified":
            self._test_connection_unified(server_ip)
            return
        
        self.log_text.append(f"\n🔍 Testing connection to {server_ip}...\n")
        self.statusBar().showMessage("Testing connection...")
        
        # Order: gpu-isolated non-unified (31883/35672), then host-network / macvlan (1883/5672)
        candidates = [
            (31883, 35672, "Docker port maps (e.g. *-gpu-isolated.yml)"),
            (1883, 5672, "Host or macvlan (brokers on standard ports)"),
        ]
        try:
            for mqtt_p, amqp_p, desc in candidates:
                if self._tcp_port_probe(server_ip, mqtt_p):
                    self.remote_mqtt_port = mqtt_p
                    self.remote_amqp_port = amqp_p
                    self.log_text.append(
                        f"✅ MQTT reachable at {server_ip}:{mqtt_p} — {desc}\n"
                    )
                    self.log_text.append(
                        f"   Client will use MQTT {mqtt_p}, AMQP {amqp_p}.\n"
                    )
                    grpc_ok = False
                    grpc_port = 50051
                    try:
                        grpc_ok = self._tcp_port_probe(server_ip, grpc_port)
                    except Exception as e:
                        self.log_text.append(f"⚠️ Could not verify gRPC port: {e}\n")
                    if grpc_ok:
                        self.log_text.append(
                            f"✅ gRPC port {grpc_port} reachable (required for unified initial model).\n"
                        )
                    else:
                        self.log_text.append(
                            f"❌ gRPC port {grpc_port} NOT reachable from this PC.\n"
                            f"   Unified clients download the first global model over gRPC; MQTT registration alone is not enough.\n"
                            f"   Open TCP {grpc_port} on the server firewall and ensure Docker maps "
                            f"\"{grpc_port}:{grpc_port}\" for the unified server.\n"
                        )
                    self.connection_status.setText("● Connected" if grpc_ok else "● Partial")
                    self.connection_status.setStyleSheet(
                        "font-size: 12px; color: #28a745;"
                        if grpc_ok
                        else "font-size: 12px; color: #e67e22;"
                    )
                    if grpc_ok:
                        QMessageBox.information(
                            self,
                            "Success",
                            f"Reachable MQTT at {server_ip}:{mqtt_p}.\n"
                            f"Using AMQP port {amqp_p}.\n"
                            f"gRPC port {grpc_port} is reachable.\n({desc})",
                        )
                    else:
                        QMessageBox.warning(
                            self,
                            "MQTT OK — gRPC blocked",
                            f"MQTT broker is reachable at {server_ip}:{mqtt_p}, but TCP port {grpc_port} (gRPC) is not.\n\n"
                            f"The unified emotion client loads the initial global model over gRPC, not over MQTT.\n"
                            f"Allow {grpc_port}/tcp on the server firewall and confirm the unified compose publishes "
                            f"{grpc_port}:{grpc_port}.",
                        )
                    self.statusBar().showMessage(
                        "Connection successful" if grpc_ok else "MQTT OK; fix gRPC port for initial model"
                    )
                    return

            self.log_text.append(
                f"❌ No MQTT on {server_ip}:31883 or :1883 (is the experiment running?)\n"
            )
            self.connection_status.setText("● Disconnected")
            self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Cannot reach MQTT at {server_ip} on ports 31883 or 1883.\n\n"
                "Check:\n"
                "• Correct server IP (e.g. eno2 address)\n"
                "• Experiment started; brokers are up (docker ps)\n"
                "• Firewall allows those TCP ports from your PC\n"
                "• Unified default bridge compose does not publish MQTT to the host — "
                "use host / macvlan mode or non-unified gpu-isolated, or add port mappings.",
            )
            self.statusBar().showMessage("Connection failed")

        except Exception as e:
            self.log_text.append(f"❌ Connection test error: {str(e)}\n")
            QMessageBox.critical(self, "Error", f"Connection test failed: {str(e)}")
            self.statusBar().showMessage("Connection test error")

    def _on_wsl2_mode_toggled(self, checked: bool) -> None:
        """Auto-fill Windows host LAN IP from the DDS peer field for this client."""
        if not checked:
            return
        try:
            cid = int(self.client_id.value())
        except (AttributeError, ValueError):
            return
        peer_map = {
            1: getattr(self, "dds_peer_client1", None),
            2: getattr(self, "dds_peer_client2", None),
            3: getattr(self, "dds_peer_client3", None),
        }
        peer_widget = peer_map.get(cid)
        if peer_widget:
            ip = peer_widget.text().strip()
            if ip and not self.wsl2_host_ip.text().strip():
                self.wsl2_host_ip.setText(ip)

    def _resolve_docker_container_name_conflict(self, container_name):
        """Free the Docker name for `docker run --name` by removing a leftover container.

        Docker errors with "already in use" if a previous run left a container with the
        same name (e.g. GUI closed without Stop, or crash). Exited containers are
        removed automatically; a running container prompts before `docker rm -f`.
        """
        try:
            r = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as e:
            self.log_text.append(f"⚠️ Could not check existing container: {e}\n")
            QMessageBox.warning(
                self,
                "Docker check failed",
                f"Could not check whether container '{container_name}' exists:\n{e}",
            )
            return False

        if r.returncode != 0:
            return True

        status = (r.stdout or "").strip().lower()
        if status == "running":
            reply = QMessageBox.question(
                self,
                "Container already running",
                f"A container named '{container_name}' is already running.\n\n"
                "Stop and remove it so a new client can start?\n\n"
                "If this session already started that client, use Stop Client first.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                return False
        else:
            self.log_text.append(
                f"Removing existing container '{container_name}' (state: {status}).\n"
            )

        rm = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if rm.returncode != 0:
            err = (rm.stderr or rm.stdout or "").strip()
            self.log_text.append(f"❌ Could not remove container: {err}\n")
            QMessageBox.critical(
                self,
                "Docker error",
                f"Could not remove container '{container_name}':\n{err}",
            )
            return False
        self.log_text.append(f"✅ Cleared old container '{container_name}'.\n")
        return True

    def start_client(self):
        """Start the distributed client container"""
        server_ip = self.server_ip.text().strip()
        client_id = self.client_id.value()
        total_clients = self.total_clients.value()
        use_case = self.use_case.currentData() or self.use_case.currentText()
        protocol = self.protocol_mode.currentData()
        network_scenario = self.network_scenario.currentData()
        data_shard = self.data_shard_combo.currentData()
        
        # Validate
        if not server_ip:
            QMessageBox.warning(self, "Error", "Please enter server IP address!")
            return
        if client_id > total_clients:
            QMessageBox.warning(
                self,
                "Invalid client count",
                f"Client ID ({client_id}) cannot be greater than Total Expected Clients ({total_clients}).\n"
                "Raise Total Expected Clients to match the main experiment (local + remote), "
                "and ensure it matches Min Clients (server) on the experiment machine.",
            )
            return

        # Determine if using unified or single protocol
        is_unified = (protocol == "rl_unified")
        
        # Confirm
        net_line = f"Network: {network_scenario}"
        if network_scenario == "dynamic":
            net_line += f" (re-randomize every {self.dynamic_interval_sec.value()}s)"
        net_line += "\n"
        shared_data_lines = (
            f"shared_data mount: {'Yes' if self.mount_shared_data.isChecked() else 'No'} "
            f"(client metrics JSONL, DBs, RL Q-tables when unified)\n"
            f"experiment_results mount: Yes → host <project>/experiment_results → /app/experiment_results "
            f"(training JSON/plots; same layout as server). Avg CPU/RAM are merged into *_training_results.json; "
            f"per-protocol charts are *_cpu_memory_per_round.png. On the experiment PC, use Experiment GUI → "
            f"Combine Plots to generate combined_cpu_memory.png across protocols.\n"
        )
        if is_unified and self.is_rl_training_mode():
            shared_data_lines += (
                f"Reset epsilon: {'Yes' if self.reset_epsilon_on_start.isChecked() else 'No (resume)'}\n"
            )
        confirm_msg = (
            f"Ready to start client with:\n\n"
            f"Client ID: {client_id}\n"
            f"Total Clients: {total_clients}\n"
            f"Server: {server_ip}\n"
            f"Use Case: {use_case}\n"
            f"Data shard: {data_shard if data_shard is not None else 'Auto (Client ID)'}\n"
            f"Protocol: {protocol}\n"
            f"RL Mode: {'Training' if self.is_rl_training_mode() else 'Inference'}\n"
            f"DDS Implementation: {self.dds_impl.currentText() if protocol == 'dds' else 'N/A'}\n"
            f"{net_line}"
            f"Rounds: {self.rounds_spinbox.value()}\n"
            f"Termination: {self.termination_mode_combo.currentText()}\n"
            f"GPU: {'Enabled' if self.gpu_enabled.isChecked() else 'Disabled'}\n"
            f"Q-Learning Convergence: {'Enabled' if self.is_rl_training_mode() else 'Disabled'}\n"
            f"Communication Model Reward: {'Enabled' if is_unified and self.communication_model_reward_enabled.isChecked() else 'Disabled'}\n"
            f"{shared_data_lines}"
            f"Quantization: {'Enabled' if self.quantization_enabled.isChecked() else 'Disabled'}\n"
            f"Compression: {'Enabled' if self.compression_enabled.isChecked() else 'Disabled'}\n"
            f"Pruning: {'Enabled' if self.pruning_enabled.isChecked() else 'Disabled'}\n\n"
            f"Continue?"
        )
        reply = QMessageBox.question(
            self,
            "Start Client",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Build docker run command
        container_name = f"fl-client-{client_id}-distributed"
        
        # Build image name - Use client-1 image as template for distributed clients
        # (The numbered images are identical, just built separately for docker-compose)
        if is_unified:
            image_name = f"docker-fl-client-unified-{use_case}-1:latest"
        else:
            image_name = f"docker-fl-client-{protocol}-{use_case}-1:latest"
        
        self.log_text.append(f"Using image: {image_name}\n")
        
        # Check if image exists locally
        try:
            check_result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                timeout=10
            )
            
            if check_result.returncode != 0:
                # Image doesn't exist
                self.log_text.append(f"⚠️ Image not found locally: {image_name}\n")
                self.log_text.append(f"The image needs to be built on this machine.\n\n")
                
                reply = QMessageBox.question(
                    self,
                    "Image Not Found",
                    f"Docker image not found on this machine:\n{image_name}\n\n"
                    f"The image needs to be built before starting the client.\n\n"
                    f"Would you like to build it now?\n"
                    f"(This may take several minutes)",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Trigger rebuild
                    self.rebuild_client_image()
                    return
                else:
                    self.log_text.append(f"❌ Cannot start client without image. Use 'Rebuild Image' button to build it.\n")
                    QMessageBox.information(
                        self,
                        "Image Required",
                        "Please use the 'Rebuild Image' button to build the Docker image first."
                    )
                    return
            else:
                self.log_text.append(f"✅ Image found: {image_name}\n")
                
        except Exception as e:
            self.log_text.append(f"⚠️ Could not check image: {str(e)}\n")
        
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        shared_data_path = os.path.join(project_root, "shared_data")
        experiment_results_host = os.path.join(project_root, "experiment_results")
        try:
            os.makedirs(experiment_results_host, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "experiment_results", f"Cannot create experiment_results directory:\n{e}")
            return
        if self.mount_shared_data.isChecked():
            try:
                os.makedirs(shared_data_path, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(self, "shared_data", f"Cannot create shared_data directory:\n{e}")
                return
            if is_unified:
                self._sync_reset_epsilon_flag_for_distributed(shared_data_path)
            eff_lbl = self._gui_resolved_network_scenario_label(network_scenario)
            self._write_current_rl_network_scenario_file(shared_data_path, eff_lbl)

        _uc_key = str(use_case).lower().replace(" ", "")
        client_use_case_env = {
            "emotion": "emotion",
            "mentalstate": "mental_state",
            "temperature": "temperature",
        }.get(_uc_key, "emotion")
        if network_scenario in (None, "none", "dynamic"):
            network_scenario_env = "default"
        else:
            network_scenario_env = str(network_scenario).strip().lower() or "default"

        # Resolve per-use-case Dataset directory so it can be bind-mounted into the
        # container, guaranteeing the CSV is present even when the image is stale.
        _uc_dir_map = {
            "emotion": "Emotion_Recognition",
            "mentalstate": "MentalState_Recognition",
            "temperature": "Temperature_Regulation",
        }
        _uc_dir = _uc_dir_map.get(_uc_key, "Temperature_Regulation")
        dataset_host_path = os.path.join(project_root, "Client", _uc_dir, "Dataset")
        dataset_container_path = f"/app/Client/{_uc_dir}/Dataset"
        dataset_mount_available = os.path.isdir(dataset_host_path)
        if dataset_mount_available:
            self.log_text.append(f"📂 Dataset mount: {dataset_host_path} → {dataset_container_path}\n")
        else:
            self.log_text.append(
                f"⚠️  Dataset directory not found on host: {dataset_host_path}\n"
                f"   The container image must contain the dataset, or training will fail.\n"
            )

        # Base command
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",  # Use host network to access server
            "--cap-add", "NET_ADMIN",   # For tc/netem network shaping
            "--cap-add", "SYS_ADMIN",   # Needed by some kernels for netem/HTB
            "-v",
            f"{experiment_results_host}:/app/experiment_results",
            "-e", f"CLIENT_ID={client_id}",
            "-e", f"NUM_CLIENTS={total_clients}",
            "-e", f"MIN_CLIENTS={total_clients}",
            "-e", f"CLIENT_USE_CASE={client_use_case_env}",
            "-e", f"NETWORK_SCENARIO={network_scenario_env}",
        ]
        # Always bind-mount the dataset directory so stale images still have the CSV.
        if dataset_mount_available:
            cmd.extend(["-v", f"{dataset_host_path}:{dataset_container_path}:ro"])
        if self.mount_shared_data.isChecked():
            cmd.extend(
                [
                    "-v",
                    f"{shared_data_path}:/shared_data",
                    "-e",
                    "CLIENT_METRICS_LOG_DIR=/shared_data",
                ]
            )
        ds_val = self.data_shard_combo.currentData()
        if ds_val is not None and str(use_case).lower() in ("emotion", "mentalstate"):
            cmd.extend(["-e", f"DATASET_CLIENT_ID={int(ds_val)}"])
        cmd.extend([
            "-e", f"NUM_ROUNDS={self.rounds_spinbox.value()}",
            "-e", f"NODE_TYPE=client",
            "-e", "CLIENT_EXPERIMENT_CHECKPOINT_EACH_ROUND=true",
            "-e", f"MQTT_BROKER={server_ip}",
            "-e", f"MQTT_PORT={self.remote_mqtt_port}",
            "-e", f"AMQP_HOST={server_ip}",
            "-e", f"AMQP_BROKER={server_ip}",
            "-e", f"AMQP_PORT={self.remote_amqp_port}",
            "-e", "AMQP_USER=guest",
            "-e", "AMQP_PASSWORD=guest",
            "-e", f"GRPC_HOST={server_ip}",
            "-e", f"GRPC_SERVER={server_ip}",
            "-e", "GRPC_PORT=50051",
            "-e", f"QUIC_HOST={server_ip}",
            "-e", "QUIC_PORT=4433",
            "-e", f"HTTP3_HOST={server_ip}",
            "-e", "HTTP3_PORT=4434",
            "-e", "DDS_DOMAIN_ID=0",
        ])

        # DDS on a remote PC: multicast LAN, or static unicast peers (DDS_PEER_*) if minimum three set.
        if protocol == "dds" or is_unified:
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _mc_xml = os.path.join(_repo_root, "config", "cyclonedds-multicast-lan.xml")
            ps = getattr(self, "dds_peer_server", None)
            p1 = getattr(self, "dds_peer_client1", None)
            p2 = getattr(self, "dds_peer_client2", None)
            p3 = getattr(self, "dds_peer_client3", None)
            ps_t = ps.text().strip() if ps else ""
            p1_t = p1.text().strip() if p1 else ""
            p2_t = p2.text().strip() if p2 else ""
            p3_t = p3.text().strip() if p3 else ""
            if ps_t and p1_t and p2_t:
                cmd.extend(
                    [
                        "-e", f"DDS_PEER_SERVER={ps_t}",
                        "-e", f"DDS_PEER_CLIENT1={p1_t}",
                        "-e", f"DDS_PEER_CLIENT2={p2_t}",
                    ]
                )
                if p3_t:
                    cmd.extend(["-e", f"DDS_PEER_CLIENT3={p3_t}"])
                self.log_text.append(
                    f"DDS: static unicast peers (SERVER={ps_t} / CLIENT1={p1_t} / CLIENT2={p2_t}"
                    + (f" / CLIENT3={p3_t}" if p3_t else "")
                    + "). CYCLONEDDS_URI is generated at runtime; open UDP between hosts "
                    "(~7400–7500 + SPDP ports 7412/7414/7416/7418). "
                    "The main experiment server must set the same DDS_PEER_* variables (no CYCLONEDDS_URI).\n"
                )
            elif os.path.isfile(_mc_xml):
                cmd.extend(
                    [
                        "-v",
                        f"{_mc_xml}:/app/config/cyclonedds-multicast-lan.xml:ro",
                        "-e",
                        "CYCLONEDDS_URI=file:///app/config/cyclonedds-multicast-lan.xml",
                    ]
                )
                self.log_text.append(
                    "DDS: CYCLONEDDS_URI=cyclonedds-multicast-lan.xml (multicast LAN). "
                    "Ensure the experiment server uses the same URI and UDP 7400–7500 is allowed.\n"
                )
            else:
                self.log_text.append(
                    f"⚠️ DDS multicast config not found ({_mc_xml}); remote DDS discovery may fail.\n"
                )

            # WSL2 NAT fix: advertise the Windows host LAN IP so remote peers can reach this client.
            # Without this, CycloneDDS advertises the internal WSL2 virtual IP (172.x.x.x) which is
            # unreachable from other LAN machines.  DDS_EXTERNAL_NETWORK_ADDRESS overrides the
            # advertised locator; DDS_NETWORK_INTERFACE pins Cyclone to the correct WSL2 interface.
            if getattr(self, "wsl2_mode", None) and self.wsl2_mode.isChecked():
                wsl2_ip = self.wsl2_host_ip.text().strip()
                wsl2_if = self.wsl2_iface.text().strip() or "eth0"
                if wsl2_ip:
                    cmd.extend(["-e", f"DDS_EXTERNAL_NETWORK_ADDRESS={wsl2_ip}"])
                    cmd.extend(["-e", f"DDS_NETWORK_INTERFACE={wsl2_if}"])
                    self.log_text.append(
                        f"WSL2 DDS fix: DDS_EXTERNAL_NETWORK_ADDRESS={wsl2_ip} "
                        f"DDS_NETWORK_INTERFACE={wsl2_if}\n"
                        "CycloneDDS will advertise the Windows host LAN IP to remote peers.\n"
                        "⚠️  Also ensure UDP ports 7412–7418 (and ~7400–7500 for RTPS user-data) are\n"
                        "   forwarded from the Windows host to the WSL2 instance.\n"
                        "   Run scripts/windows_wsl2_dds_portforward.ps1 on Windows, or enable\n"
                        "   WSL2 mirrored networking (Windows 11 22H2+) in %USERPROFILE%\\.wslconfig.\n"
                    )
                else:
                    self.log_text.append(
                        "⚠️ WSL2 mode is checked but 'Windows host LAN IP' is empty. "
                        "DDS_EXTERNAL_NETWORK_ADDRESS was NOT set; DDS discovery may fail.\n"
                    )

        termination_mode = self.termination_mode_combo.currentData() or self.termination_mode_combo.currentText()
        stop_on_client_convergence = "false" if termination_mode == "fixed_rounds" else "true"
        cmd.extend(["-e", f"STOP_ON_CLIENT_CONVERGENCE={stop_on_client_convergence}"])
        cmd.extend([
            "-e",
            f"TRAINING_TERMINATION_MODE={'fixed_rounds' if termination_mode == 'fixed_rounds' else 'client_convergence'}",
        ])
        
        # DDS implementation vendor selection (passed as environment variable)
        if protocol == "dds":
            dds_impl_value = self.dds_impl.currentData() or self.dds_impl.currentText()
            cmd.extend(["-e", f"DDS_IMPL={dds_impl_value}"])
        
        # Add unified-specific env vars
        if is_unified:
            cmd.extend([
                "-e", "USE_RL_SELECTION=true",
                "-e", "ENABLE_METRICS=true",
                "-e", f"USE_COMMUNICATION_MODEL_REWARD={'true' if self.communication_model_reward_enabled.isChecked() else 'false'}",
            ])
            if self.is_rl_training_mode():
                cmd.extend([
                    "-e", "USE_QL_CONVERGENCE=true",
                    "-e", "USE_RL_EXPLORATION=true",
                    "-e", "Q_CONVERGENCE_THRESHOLD=0.01",
                    "-e", "Q_CONVERGENCE_PATIENCE=5",
                    "-e", "RL_BOUNDARY_PIPELINE=true",
                    "-e", "RL_PHASE0_ROUNDS=20",
                    "-e", f"RESET_EPSILON={'true' if self.reset_epsilon_on_start.isChecked() else 'false'}",
                ])
            else:
                cmd.extend([
                    "-e", "USE_QL_CONVERGENCE=false",
                    "-e", "USE_RL_EXPLORATION=false",
                    "-e", "RESET_EPSILON=false",
                ])
        else:
            cmd.extend([
                "-e", f"PROTOCOL={protocol.upper()}"
            ])

        if self.quantization_enabled.isChecked():
            cmd.extend([
                "-e", "USE_QUANTIZATION=true",
                "-e", f"QUANTIZATION_BITS={self.quant_bits.currentText()}",
                "-e", f"QUANTIZATION_STRATEGY={self.quant_strategy.currentText()}",
                "-e", f"QUANTIZATION_SYMMETRIC={'true' if self.quant_symmetric.isChecked() else 'false'}",
                "-e", f"QUANTIZATION_PER_CHANNEL={'true' if self.quant_per_channel.isChecked() else 'false'}",
            ])
        if self.pruning_enabled.isChecked():
            cmd.extend([
                "-e", "USE_PRUNING=true",
                "-e", f"PRUNING_SPARSITY={self.pruning_ratio.value() / 100.0:.2f}",
            ])
        if self.compression_enabled.isChecked():
            cmd.extend([
                "-e", "ENABLE_COMPRESSION=true",
                "-e", f"COMPRESSION_ALGORITHM={self.compression_algo.currentText()}",
                "-e", f"COMPRESSION_LEVEL={self.compression_level.value()}",
            ])

        cmd.extend([
            "-e", f"SAVE_CHECKPOINTS={'true' if self.save_checkpoints.isChecked() else 'false'}",
            "-e", f"VERBOSE_LOGGING={'true' if self.verbose_logging.isChecked() else 'false'}",
            "-e", f"ENABLE_TENSORBOARD={'true' if self.enable_tensorboard.isChecked() else 'false'}",
            "-e", f"PROFILE_PERFORMANCE={'true' if self.profile_performance.isChecked() else 'false'}",
        ])
        
        # GPU support
        if self.gpu_enabled.isChecked():
            cmd.extend(["--gpus", "all"])
        
        # Add image and entrypoint (must match compose `command`; Dockerfile default is MQTT-only)
        cmd.append(image_name)
        cmd.extend(self._docker_client_entrypoint(use_case, protocol, is_unified))

        if not self._resolve_docker_container_name_conflict(container_name):
            return

        self.log_text.append(f"\n🚀 Starting client container...\n")
        self.log_text.append(f"Command: {' '.join(cmd)}\n\n")
        
        try:
            # Start container
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.client_container = container_name
                self.log_text.append(f"✅ Client container started: {container_name}\n")
                
                # Apply network conditions if specified
                if network_scenario != "none":
                    self.apply_network_conditions(container_name, network_scenario)
                if self.mount_shared_data.isChecked():
                    eff_lbl2 = self._gui_resolved_network_scenario_label(network_scenario)
                    self._write_current_rl_network_scenario_file(shared_data_path, eff_lbl2)
                if network_scenario == "dynamic":
                    self._dynamic_network_timer.start(self.dynamic_interval_sec.value() * 1000)
                
                # Update UI
                self.network_scenario.setEnabled(False)
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self._refresh_dynamic_randomize_btn()
                self.status_label.setText(f"Status: Running (Client {client_id})")
                self.status_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #28a745;")
                self.statusBar().showMessage(f"Client {client_id} running")
                
                # Start monitoring
                self.start_monitoring(container_name)
                
            else:
                self.log_text.append(f"❌ Failed to start container:\n{result.stderr}\n")
                QMessageBox.critical(self, "Error", f"Failed to start client:\n{result.stderr}")
                
        except Exception as e:
            self.log_text.append(f"❌ Error: {str(e)}\n")
            QMessageBox.critical(self, "Error", f"Error starting client: {str(e)}")
    
    def apply_network_conditions(self, container_name, scenario):
        """Apply network conditions to client container"""
        if scenario == "none" and not self.enable_congestion.isChecked():
            return

        cond = self.get_effective_network_profile()
        self.log_text.append(f"\n🌐 Applying network conditions: {scenario}\n")
        self.log_text.append(
            f"Resolved profile -> latency={cond['latency']}ms, bandwidth={cond['bandwidth']}mbit, "
            f"jitter={cond['jitter']}ms, loss={cond['packet_loss']:.1f}%\n"
        )
        self._apply_tc_profile(container_name, cond)

    def _parse_dev_from_ip_route_text(self, text: str):
        """Return first non-loopback `dev` from `ip route` output, or None."""
        if not text or not text.strip():
            return None
        first = text.strip().splitlines()[0]
        m = _IP_ROUTE_DEV_RE.search(first)
        if not m:
            return None
        dev = m.group(1)
        return None if dev == "lo" else dev

    def _route_iface_on_host(self, dest_ip: str):
        """Outgoing iface toward dest on this machine (same view as --network host)."""
        try:
            r = subprocess.run(
                ["ip", "-4", "route", "get", dest_ip],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if r.returncode == 0:
                return self._parse_dev_from_ip_route_text(r.stdout)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return None

    def _host_default_route_iface(self):
        try:
            r = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return self._parse_dev_from_ip_route_text(r.stdout)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return None

    def _route_iface_docker_exec(self, container_name: str, dest_ip: str):
        """Same as host when using network_mode=host; kept as secondary probe."""
        try:
            r = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "/bin/sh",
                    "-c",
                    f"ip -4 route get {dest_ip} 2>/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if r.returncode == 0:
                return self._parse_dev_from_ip_route_text(r.stdout)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return None

    def _iface_exists_in_container_netns(self, container_name: str, iface: str) -> bool:
        if not iface:
            return False
        try:
            r = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "/bin/sh",
                    "-c",
                    f"test -e /sys/class/net/{iface}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _tc_netdev_for_container(self, container_name: str) -> str:
        """Interface to shape for --network host (real NIC is often enp*/wlan*, not eth0)."""
        server_ip = self.server_ip.text().strip()
        if not server_ip:
            return "eth0"
        # Prefer host `ip route get`: with Docker host networking, container and host share
        # the same routing table; this avoids relying on docker exec/bash when resolving iface.
        ordered = []
        for iface in (
            self._route_iface_on_host(server_ip),
            self._route_iface_docker_exec(container_name, server_ip),
            self._host_default_route_iface(),
        ):
            if iface and iface not in ordered:
                ordered.append(iface)
        for iface in ordered:
            if self._iface_exists_in_container_netns(container_name, iface):
                return iface
        if ordered:
            return ordered[0]
        return "eth0"

    def _apply_tc_profile(self, container_name, cond):
        """Apply tc/netem profile inside the client container (detect iface toward server IP)."""
        try:
            # Ensure sch_netem and sch_htb kernel modules are available on the host
            # (needed for --network host containers; failures are non-fatal).
            for mod in ("sch_netem", "sch_htb"):
                try:
                    subprocess.run(
                        ["sudo", "modprobe", mod],
                        capture_output=True, timeout=5,
                    )
                except (subprocess.SubprocessError, FileNotFoundError, OSError):
                    pass

            dev = self._tc_netdev_for_container(container_name)
            self.log_text.append(f"  tc netdev: {dev} (route toward server {self.server_ip.text().strip()})\n")
            setup_cmds = [
                f"tc qdisc del dev {dev} root || true",
                f"tc qdisc add dev {dev} root handle 1: htb default 12",
                f"tc class add dev {dev} parent 1: classid 1:12 htb rate {cond['bandwidth']}mbit",
            ]

            netem_params = [f"delay {cond['latency']}ms"]
            if cond["jitter"] > 0:
                netem_params.append(f"{cond['jitter']}ms")
            if cond["packet_loss"] > 0:
                netem_params.append(f"loss {cond['packet_loss']:.1f}%")

            setup_cmds.append(
                f"tc qdisc add dev {dev} parent 1:12 handle 10: netem {' '.join(netem_params)}"
            )

            all_ok = True
            for tc_cmd in setup_cmds:
                result = subprocess.run(
                    ["docker", "exec", container_name, "/bin/sh", "-c", tc_cmd],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    all_ok = False
                    err = (result.stderr or result.stdout or "").strip()
                    self.log_text.append(f"⚠️ Warning applying network condition: {err}\n")

            if all_ok:
                self.log_text.append("✅ Network conditions applied successfully\n")
            else:
                self.log_text.append(
                    "❌ Network shaping did not apply cleanly. "
                    "Check that the client container has CAP_NET_ADMIN, `iproute2`/`tc` inside the image, "
                    "and that the interface name above exists in the container "
                    "(on host networking, run `ip -4 route get <SERVER_IP>` on the host to see the correct NIC).\n"
                )

        except Exception as e:
            self.log_text.append(f"⚠️ Error applying network conditions: {str(e)}\n")
    
    def start_monitoring(self, container_name):
        """Start monitoring client logs"""
        self.client_monitor = ClientMonitor(container_name)
        self.client_monitor.log_update.connect(self.update_logs)
        self.client_monitor.start()
    
    def _on_dynamic_interval_changed(self, _value):
        if self._dynamic_network_timer.isActive():
            self._dynamic_network_timer.start(self.dynamic_interval_sec.value() * 1000)

    def _on_dynamic_network_tick(self):
        if not self.client_container:
            self._dynamic_network_timer.stop()
            return
        if self.network_scenario.currentData() != "dynamic":
            self._dynamic_network_timer.stop()
            return
        self.apply_random_dynamic_base()
        cond = self.get_effective_network_profile()
        self.log_text.append(
            f"\n🌐 [DYNAMIC] Switched to {self._last_dynamic_base}: "
            f"lat={cond['latency']}ms, bw={cond['bandwidth']}Mbps, "
            f"jitter={cond['jitter']}ms, loss={cond['packet_loss']:.1f}%\n"
        )
        self._apply_tc_profile(self.client_container, cond)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        shared_data_path = os.path.join(project_root, "shared_data")
        if self.mount_shared_data.isChecked() and os.path.isdir(shared_data_path):
            eff = self._gui_resolved_network_scenario_label("dynamic")
            self._write_current_rl_network_scenario_file(shared_data_path, eff)

    def on_randomize_dynamic_now(self):
        if not self.client_container:
            return
        if self.network_scenario.currentData() != "dynamic":
            return
        self.apply_random_dynamic_base()
        cond = self.get_effective_network_profile()
        self.log_text.append(f"\n🌐 [DYNAMIC] Manual randomize → {self._last_dynamic_base}\n")
        self._apply_tc_profile(self.client_container, cond)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        shared_data_path = os.path.join(project_root, "shared_data")
        if self.mount_shared_data.isChecked() and os.path.isdir(shared_data_path):
            eff = self._gui_resolved_network_scenario_label("dynamic")
            self._write_current_rl_network_scenario_file(shared_data_path, eff)

    def _refresh_dynamic_randomize_btn(self):
        running = self.client_container is not None
        dyn = self.network_scenario.currentData() == "dynamic"
        self.btn_dynamic_randomize.setEnabled(running and dyn)

    def stop_client(self):
        """Stop the client container"""
        if not self.client_container:
            return
        
        reply = QMessageBox.question(
            self,
            "Stop Client",
            "Are you sure you want to stop the client?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        try:
            self._dynamic_network_timer.stop()
            # Stop monitoring
            if self.client_monitor:
                self.client_monitor.stop()
                self.client_monitor.wait()
                self.client_monitor = None
            
            # Stop and remove container
            self.log_text.append(f"\n⏹️ Stopping client container...\n")
            subprocess.run(["docker", "stop", self.client_container], timeout=30)
            subprocess.run(["docker", "rm", self.client_container], timeout=30)
            self.log_text.append(f"✅ Client stopped and removed\n")
            
            # Reset UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Status: Not Running")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #6c757d;")
            self.statusBar().showMessage("Client stopped")
            self.client_container = None
            self.network_scenario.setEnabled(True)
            self._refresh_dynamic_randomize_btn()
            
        except Exception as e:
            self.log_text.append(f"❌ Error stopping client: {str(e)}\n")
            QMessageBox.critical(self, "Error", f"Error stopping client: {str(e)}")
    
    def update_logs(self, text):
        """Update log display"""
        self.log_text.insertPlainText(text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def rebuild_client_image(self):
        """Rebuild the client Docker image for current configuration"""
        use_case = self.use_case.currentData() or self.use_case.currentText()
        protocol = self.protocol_mode.currentData()
        
        # Determine if using unified or single protocol
        is_unified = (protocol == "rl_unified")
        
        use_case_file = str(use_case).lower().replace(" ", "")
        
        # Map to docker-compose file and service pattern
        if is_unified:
            compose_file = f"docker-compose-unified-{use_case_file}.yml"
            # Build all unified client services (they share the same image)
            service_pattern = f"fl-client-unified-{use_case_file}-"
        else:
            compose_file = f"docker-compose-{use_case_file}.yml"
            # Build all client services for this protocol (they share the same image)
            service_pattern = f"fl-client-{protocol}-{use_case_file}-"
        
        # Confirm rebuild
        reply = QMessageBox.question(
            self,
            "Rebuild Image",
            f"This will rebuild the Docker client images:\n\n"
            f"Use Case: {use_case}\n"
            f"Protocol: {protocol}\n"
            f"Compose File: {compose_file}\n"
            f"Services: {service_pattern}*\n\n"
            f"This may take several minutes.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Get Docker directory path (assuming standard project structure)
        docker_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Docker'))
        
        if not os.path.exists(os.path.join(docker_dir, compose_file)):
            QMessageBox.critical(
                self,
                "Error",
                f"Docker compose file not found:\n{compose_file}\n\n"
                f"Expected location: {docker_dir}"
            )
            return
        
        self.log_text.append(f"\n{'='*60}\n")
        self.log_text.append(f"🔨 Rebuilding Docker client images...\n")
        self.log_text.append(f"Directory: {docker_dir}\n")
        self.log_text.append(f"Compose File: {compose_file}\n")
        self.log_text.append(f"Service Pattern: {service_pattern}*\n")
        self.log_text.append(f"{'='*60}\n\n")
        
        # Disable buttons during rebuild
        self.rebuild_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.rebuild_btn.setText("⏳ Building...")
        
        # Run docker compose build in background
        def run_rebuild():
            try:
                # Build all client services matching the pattern
                # This ensures all client-1, client-2, etc. are rebuilt with same image
                cmd = ["docker", "compose", "-f", compose_file, "build", "--parallel"]
                
                # Add all matching service names
                # Read compose file to find matching services
                import yaml
                compose_path = os.path.join(docker_dir, compose_file)
                try:
                    with open(compose_path, 'r') as f:
                        compose_data = yaml.safe_load(f)
                        services = compose_data.get('services', {})
                        matching_services = [s for s in services.keys() if s.startswith(service_pattern)]
                        
                        if matching_services:
                            cmd.extend(matching_services)
                            self.log_text.append(f"Building services: {', '.join(matching_services)}\n\n")
                        else:
                            self.log_text.append(f"⚠️ No services found matching pattern: {service_pattern}*\n")
                            self.log_text.append(f"Available services: {', '.join(services.keys())}\n")
                            raise ValueError(f"No matching services found for {service_pattern}*")
                except Exception as e:
                    self.log_text.append(f"Error reading compose file: {e}\n")
                    self.log_text.append(f"Attempting to build with pattern anyway...\n")
                    # Fallback: try building first two clients
                    cmd.extend([f"{service_pattern}1", f"{service_pattern}2"])
                
                process = subprocess.Popen(
                    cmd,
                    cwd=docker_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output
                for line in process.stdout:
                    self.log_text.append(line.rstrip())
                    self.log_text.verticalScrollBar().setValue(
                        self.log_text.verticalScrollBar().maximum()
                    )
                
                process.wait()
                
                if process.returncode == 0:
                    self.log_text.append(f"\n✅ Build completed successfully!\n")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Docker image rebuilt successfully!\n\n"
                        f"You can now start the client with the updated code."
                    )
                else:
                    self.log_text.append(f"\n❌ Build failed with exit code {process.returncode}\n")
                    QMessageBox.critical(
                        self,
                        "Build Failed",
                        f"Docker build failed with exit code {process.returncode}\n\n"
                        f"Check the logs for details."
                    )
                    
            except Exception as e:
                self.log_text.append(f"\n❌ Error during build: {str(e)}\n")
                QMessageBox.critical(self, "Error", f"Build error:\n{str(e)}")
            finally:
                # Re-enable buttons
                self.rebuild_btn.setEnabled(True)
                self.start_btn.setEnabled(True)
                self.rebuild_btn.setText("🔨 Rebuild Image")
        
        # Run in background thread
        thread = threading.Thread(target=run_rebuild, daemon=True)
        thread.start()
    
    def reset_databases(self):
        """Reset/delete all database files"""
        reply = QMessageBox.question(
            self,
            "Reset Databases",
            "This will delete ALL database files:\n\n"
            "- packet_logs_server.db\n"
            "- packet_logs_client_*.db\n"
            "- q_learning_client_*.db\n\n"
            "This action cannot be undone!\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Find shared_data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shared_data_dir = os.path.join(os.path.dirname(script_dir), 'shared_data')
        
        if not os.path.exists(shared_data_dir):
            QMessageBox.warning(
                self,
                "Directory Not Found",
                f"shared_data directory not found at:\n{shared_data_dir}\n\n"
                f"Databases may not have been created yet."
            )
            return
        
        deleted_files = []
        failed_files = []
        
        # Find and delete all database files
        db_patterns = [
            os.path.join(shared_data_dir, 'packet_logs_*.db'),
            os.path.join(shared_data_dir, 'q_learning_*.db'),
            os.path.join(shared_data_dir, 'q_learning.db'),
            os.path.join(shared_data_dir, 'packet_logs.db')
        ]
        
        for pattern in db_patterns:
            for db_file in glob.glob(pattern):
                try:
                    os.remove(db_file)
                    deleted_files.append(os.path.basename(db_file))
                    self.log_text.append(f"✅ Deleted: {os.path.basename(db_file)}\n")
                except Exception as e:
                    failed_files.append((os.path.basename(db_file), str(e)))
                    self.log_text.append(f"❌ Failed to delete {os.path.basename(db_file)}: {str(e)}\n")
        
        if deleted_files:
            QMessageBox.information(
                self,
                "Databases Reset",
                f"Successfully deleted {len(deleted_files)} database file(s):\n\n"
                + "\n".join(deleted_files[:10])  # Show first 10 files
                + (f"\n... and {len(deleted_files) - 10} more" if len(deleted_files) > 10 else "")
            )
            self.log_text.append(f"\n✅ Database reset complete. Deleted {len(deleted_files)} file(s).\n\n")
        else:
            QMessageBox.information(
                self,
                "No Databases Found",
                "No database files found to delete.\n\n"
                "Databases are created when containers start running."
            )
        
        if failed_files:
            QMessageBox.warning(
                self,
                "Some Files Failed",
                f"Failed to delete {len(failed_files)} file(s).\n\n"
                "Check logs for details."
            )
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.client_container:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Client is still running. Stop it before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.Yes:
                self.stop_client()
        
        # Stop monitoring
        if self.client_monitor:
            self.client_monitor.stop()
            self.client_monitor.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show GUI
    gui = DistributedClientGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
