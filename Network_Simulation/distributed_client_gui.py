#!/usr/bin/env python3
"""
Distributed Client GUI for Federated Learning
Allows running clients on separate PCs that connect to a central experiment server
"""

import sys
import os
import json
import subprocess
import threading
import glob
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QSlider,
    QGroupBox, QTextEdit, QProgressBar, QGridLayout,
    QMessageBox, QLineEdit, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor


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
        self.init_ui()
        
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
        
        # Connection Configuration
        connection_group = self.create_connection_config()
        main_layout.addWidget(connection_group)
        
        # Client Configuration
        client_config_group = self.create_client_config()
        main_layout.addWidget(client_config_group)
        
        # Network Scenario Selection
        network_group = self.create_network_config()
        main_layout.addWidget(network_group)
        
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
    
    def create_connection_config(self):
        """Create connection configuration section"""
        group = QGroupBox("🔌 Server Connection Settings")
        group.setStyleSheet(self.get_group_style())
        layout = QGridLayout()
        
        # Server IP
        layout.addWidget(QLabel("Server IP Address:"), 0, 0)
        self.server_ip = QLineEdit("192.168.1.100")
        self.server_ip.setPlaceholderText("e.g., 192.168.1.100")
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
            "Standard Ports: MQTT(1883) | AMQP(5672) | gRPC(50051) | QUIC(4433) | DDS(Domain 0)"
        )
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(info_label, 1, 0, 1, 3)
        
        group.setLayout(layout)
        return group
    
    def create_client_config(self):
        """Create client configuration section"""
        group = QGroupBox("⚙️ Client Configuration")
        group.setStyleSheet(self.get_group_style())
        layout = QGridLayout()
        
        # Client ID
        layout.addWidget(QLabel("Client ID:"), 0, 0)
        self.client_id = QSpinBox()
        self.client_id.setRange(1, 100)
        self.client_id.setValue(3)
        self.client_id.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.client_id, 0, 1)
        
        # Use Case
        layout.addWidget(QLabel("Use Case:"), 1, 0)
        self.use_case = QComboBox()
        self.use_case.addItems(["emotion", "mentalstate", "temperature"])
        self.use_case.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.use_case, 1, 1)
        
        # Protocol Selection Mode
        layout.addWidget(QLabel("Protocol Mode:"), 2, 0)
        self.protocol_mode = QComboBox()
        self.protocol_mode.addItem("RL-Unified (Auto Select)", "rl_unified")
        self.protocol_mode.addItem("MQTT", "mqtt")
        self.protocol_mode.addItem("AMQP", "amqp")
        self.protocol_mode.addItem("gRPC", "grpc")
        self.protocol_mode.addItem("QUIC", "quic")
        self.protocol_mode.addItem("DDS", "dds")
        self.protocol_mode.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(self.protocol_mode, 2, 1)
        
        # GPU Support
        self.gpu_enabled = QCheckBox("Enable GPU (if available)")
        self.gpu_enabled.setStyleSheet("padding: 5px; font-size: 12px;")
        layout.addWidget(self.gpu_enabled, 3, 0, 1, 2)
        
        # Q-Learning Convergence Training (only for RL-Unified)
        self.ql_convergence_enabled = QCheckBox("Train until Q-value converges (RL-Unified only)")
        self.ql_convergence_enabled.setStyleSheet("padding: 5px; font-size: 12px;")
        self.ql_convergence_enabled.setToolTip("When enabled, training continues until Q-learning values converge instead of accuracy-based convergence")
        layout.addWidget(self.ql_convergence_enabled, 4, 0, 1, 2)
        
        # Update checkbox state when protocol mode changes
        self.protocol_mode.currentIndexChanged.connect(self.update_ql_convergence_visibility)
        self.update_ql_convergence_visibility()
        
        group.setLayout(layout)
        return group
    
    def create_network_config(self):
        """Create network scenario configuration"""
        group = QGroupBox("🌐 Network Conditions (Applied to This Client)")
        group.setStyleSheet(self.get_group_style())
        layout = QVBoxLayout()
        
        # Preset scenarios
        scenario_layout = QHBoxLayout()
        scenario_layout.addWidget(QLabel("Preset Scenario:"))
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
        self.network_scenario.setStyleSheet("padding: 8px; font-size: 12px;")
        self.network_scenario.currentIndexChanged.connect(self.update_network_preview)
        scenario_layout.addWidget(self.network_scenario)
        scenario_layout.addStretch()
        layout.addLayout(scenario_layout)
        
        # Network preview
        self.network_preview = QLabel("No network simulation")
        self.network_preview.setStyleSheet("""
            background-color: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 4px;
            padding: 10px;
            font-size: 11px;
            color: #495057;
        """)
        layout.addWidget(self.network_preview)
        
        group.setLayout(layout)
        return group
    
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
        self.ql_convergence_enabled.setEnabled(is_rl_unified)
        if not is_rl_unified:
            self.ql_convergence_enabled.setChecked(False)
    
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
        scenarios = {
            "none": "No network simulation applied",
            "excellent": "Latency: 5ms | Bandwidth: 100Mbps | Loss: 0%",
            "good": "Latency: 20ms | Bandwidth: 50Mbps | Loss: 0.1%",
            "moderate": "Latency: 50ms | Bandwidth: 20Mbps | Loss: 0.5%",
            "poor": "Latency: 100ms | Bandwidth: 5Mbps | Loss: 1%",
            "very_poor": "Latency: 200ms | Bandwidth: 1Mbps | Loss: 3%",
            "satellite": "Latency: 600ms | Bandwidth: 10Mbps | Jitter: 50ms | Loss: 2%",
            "congested_light": "Latency: 30ms | Bandwidth: 30Mbps | Jitter: 5ms | Loss: 0.5%",
            "congested_moderate": "Latency: 75ms | Bandwidth: 15Mbps | Jitter: 15ms | Loss: 1.5%",
            "congested_heavy": "Latency: 150ms | Bandwidth: 5Mbps | Jitter: 30ms | Loss: 3%"
        }
        
        scenario = self.network_scenario.currentData()
        preview_text = scenarios.get(scenario, "Unknown scenario")
        self.network_preview.setText(f"📊 {preview_text}")
    
    def test_connection(self):
        """Test connection to server"""
        server_ip = self.server_ip.text().strip()
        
        if not server_ip:
            QMessageBox.warning(self, "Error", "Please enter server IP address!")
            return
        
        self.log_text.append(f"\n🔍 Testing connection to {server_ip}...\n")
        self.statusBar().showMessage("Testing connection...")
        
        # Test MQTT port
        try:
            result = subprocess.run(
                ["timeout", "3", "bash", "-c", f"nc -zv {server_ip} 31883"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.log_text.append(f"✅ MQTT broker reachable at {server_ip}:31883\n")
                self.connection_status.setText("● Connected")
                self.connection_status.setStyleSheet("font-size: 12px; color: #28a745;")
                QMessageBox.information(self, "Success", f"Successfully connected to server at {server_ip}!")
                self.statusBar().showMessage("Connection successful")
            else:
                self.log_text.append(f"❌ Cannot reach MQTT broker at {server_ip}:31883\n")
                self.connection_status.setText("● Disconnected")
                self.connection_status.setStyleSheet("font-size: 12px; color: #dc3545;")
                QMessageBox.warning(self, "Connection Failed", 
                                   f"Cannot reach server at {server_ip}!\n\n"
                                   f"Please check:\n"
                                   f"• Server IP is correct\n"
                                   f"• Server is running\n"
                                   f"• Network connectivity\n"
                                   f"• Firewall settings\n"
                                   f"• MQTT broker listening on port 31883")
                self.statusBar().showMessage("Connection failed")
                
        except Exception as e:
            self.log_text.append(f"❌ Connection test error: {str(e)}\n")
            QMessageBox.critical(self, "Error", f"Connection test failed: {str(e)}")
            self.statusBar().showMessage("Connection test error")
    
    def start_client(self):
        """Start the distributed client container"""
        server_ip = self.server_ip.text().strip()
        client_id = self.client_id.value()
        use_case = self.use_case.currentText()
        protocol = self.protocol_mode.currentData()
        network_scenario = self.network_scenario.currentData()
        
        # Validate
        if not server_ip:
            QMessageBox.warning(self, "Error", "Please enter server IP address!")
            return
        
        # Determine if using unified or single protocol
        is_unified = (protocol == "rl_unified")
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Start Client",
            f"Ready to start client with:\n\n"
            f"Client ID: {client_id}\n"
            f"Server: {server_ip}\n"
            f"Use Case: {use_case}\n"
            f"Protocol: {protocol}\n"
            f"Network: {network_scenario}\n"
            f"GPU: {'Enabled' if self.gpu_enabled.isChecked() else 'Disabled'}\n"
            f"Q-Learning Convergence: {'Enabled' if (is_unified and self.ql_convergence_enabled.isChecked()) else 'Disabled'}\n\n"
            f"Continue?",
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
        
        # Base command
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "host",  # Use host network to access server
            "--cap-add", "NET_ADMIN",  # For network simulation
            "-e", f"CLIENT_ID={client_id}",
            "-e", f"NODE_TYPE=client",
            "-e", f"MQTT_BROKER={server_ip}",
            "-e", "MQTT_PORT=31883",  # External port for MQTT broker
            "-e", f"AMQP_BROKER={server_ip}",
            "-e", "AMQP_PORT=35672",  # External port for RabbitMQ broker
            "-e", "AMQP_USER=guest",
            "-e", "AMQP_PASSWORD=guest",
            "-e", f"GRPC_SERVER={server_ip}",
            "-e", "GRPC_PORT=50051",
            "-e", f"QUIC_HOST={server_ip}",
            "-e", "QUIC_PORT=4433",
            "-e", "DDS_DOMAIN_ID=0"
        ]
        
        # Add unified-specific env vars
        if is_unified:
            cmd.extend([
                "-e", "USE_RL_SELECTION=true",
                "-e", "ENABLE_METRICS=true"
            ])
            # Add Q-learning convergence option if enabled
            if self.ql_convergence_enabled.isChecked():
                cmd.extend([
                    "-e", "USE_QL_CONVERGENCE=true",
                    "-e", "Q_CONVERGENCE_THRESHOLD=0.01",
                    "-e", "Q_CONVERGENCE_PATIENCE=5"
                ])
        else:
            cmd.extend([
                "-e", f"PROTOCOL={protocol.upper()}"
            ])
        
        # GPU support
        if self.gpu_enabled.isChecked():
            cmd.extend(["--gpus", "all"])
        
        # Add image
        cmd.append(image_name)
        
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
                
                # Update UI
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
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
        conditions = {
            "excellent": {"latency": "5ms", "bandwidth": "100mbit", "loss": "0"},
            "good": {"latency": "20ms", "bandwidth": "50mbit", "loss": "0.1"},
            "moderate": {"latency": "50ms", "bandwidth": "20mbit", "loss": "0.5"},
            "poor": {"latency": "100ms", "bandwidth": "5mbit", "loss": "1"},
            "very_poor": {"latency": "200ms", "bandwidth": "1mbit", "loss": "3"},
            "satellite": {"latency": "600ms", "bandwidth": "10mbit", "jitter": "50ms", "loss": "2"},
            "congested_light": {"latency": "30ms", "bandwidth": "30mbit", "jitter": "5ms", "loss": "0.5"},
            "congested_moderate": {"latency": "75ms", "bandwidth": "15mbit", "jitter": "15ms", "loss": "1.5"},
            "congested_heavy": {"latency": "150ms", "bandwidth": "5mbit", "jitter": "30ms", "loss": "3"}
        }
        
        if scenario not in conditions:
            return
        
        cond = conditions[scenario]
        self.log_text.append(f"\n🌐 Applying network conditions: {scenario}\n")
        
        try:
            # Build tc commands to apply inside container
            setup_cmds = [
                "tc qdisc add dev eth0 root handle 1: htb default 12",
                f"tc class add dev eth0 parent 1: classid 1:12 htb rate {cond['bandwidth']}",
            ]
            
            netem_params = [f"delay {cond['latency']}"]
            if 'jitter' in cond:
                netem_params.append(cond['jitter'])
            if 'loss' in cond and float(cond['loss']) > 0:
                netem_params.append(f"loss {cond['loss']}%")
            
            setup_cmds.append(f"tc qdisc add dev eth0 parent 1:12 handle 10: netem {' '.join(netem_params)}")
            
            for tc_cmd in setup_cmds:
                result = subprocess.run(
                    ["docker", "exec", container_name, "bash", "-c", tc_cmd],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    self.log_text.append(f"⚠️ Warning applying network condition: {result.stderr}\n")
            
            self.log_text.append(f"✅ Network conditions applied successfully\n")
            
        except Exception as e:
            self.log_text.append(f"⚠️ Error applying network conditions: {str(e)}\n")
    
    def start_monitoring(self, container_name):
        """Start monitoring client logs"""
        self.client_monitor = ClientMonitor(container_name)
        self.client_monitor.log_update.connect(self.update_logs)
        self.client_monitor.start()
    
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
        use_case = self.use_case.currentText()
        protocol = self.protocol_mode.currentData()
        
        # Determine if using unified or single protocol
        is_unified = (protocol == "rl_unified")
        
        # Map use case display name to file name
        use_case_map = {
            "Emotion Recognition": "emotion",
            "Mental State Recognition": "mentalstate",
            "Temperature Regulation": "temperature"
        }
        use_case_file = use_case_map.get(use_case, use_case.lower().replace(" ", ""))
        
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
