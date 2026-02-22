#!/usr/bin/env python3
"""
Federated Learning Network Experiment GUI
A beautiful and comprehensive interface for running FL experiments
"""

import sys
import os
import json
import subprocess
import threading
from datetime import datetime

# Add GUI directory to path for packet_logs_tab import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'GUI'))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox, QSlider,
    QGroupBox, QTextEdit, QProgressBar, QTabWidget, QGridLayout,
    QScrollArea, QMessageBox, QLineEdit, QRadioButton, QButtonGroup,
    QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon


class DashboardMonitor(QThread):
    """Background thread for FL training dashboard monitoring"""
    dashboard_update = pyqtSignal(str)
    
    def __init__(self, use_case, parent=None):
        super().__init__(parent)
        self.use_case = use_case
        self.running = True
        self.process = None
        
    def run(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dashboard_script = os.path.join(script_dir, "fl_training_dashboard.py")
            
            self.process = subprocess.Popen(
                ["python3", dashboard_script, "--use-case", self.use_case, "--interval", "5"],
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
                    self.dashboard_update.emit(line)
                    
        except Exception as e:
            self.dashboard_update.emit(f"Dashboard Error: {str(e)}\n")
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


class LogMonitor(QThread):
    """Background thread for monitoring container logs"""
    log_update = pyqtSignal(str, str)  # (log_type, message)
    
    def __init__(self, container_name, log_type, parent=None):
        super().__init__(parent)
        self.container_name = container_name
        self.log_type = log_type
        self.running = True
        self.process = None
        
    def run(self):
        try:
            self.process = subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "50", self.container_name],
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
                    self.log_update.emit(self.log_type, line)
                    
        except Exception as e:
            self.log_update.emit(self.log_type, f"Log Error: {str(e)}\n")
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


class NetworkController(QThread):
    """Background thread for network control via fl_network_monitor"""
    control_update = pyqtSignal(str)
    
    def __init__(self, latency=0, bandwidth=0, jitter=0, packet_loss=0, target="all", parent=None):
        super().__init__(parent)
        self.latency = latency
        self.bandwidth = bandwidth
        self.jitter = jitter
        self.packet_loss = packet_loss
        self.target = target  # "all", "server", or specific client container name
        
    def run(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            monitor_script = os.path.join(script_dir, "fl_network_monitor.py")
            
            # Build command based on target
            if self.target == "all":
                cmd = ["python3", monitor_script, "--all"]
            elif self.target == "server":
                cmd = ["python3", monitor_script, "--server"]
            else:
                # Specific client container - extract client ID
                # Assumes container names like "client_1", "client_2", etc.
                client_id = self.target.split('_')[-1] if '_' in self.target else "1"
                cmd = ["python3", monitor_script, "--client-id", client_id]
            
            if self.latency > 0:
                cmd.extend(["--latency", f"{self.latency}ms"])
            if self.bandwidth > 0:
                cmd.extend(["--bandwidth", f"{self.bandwidth}mbit"])
            if self.jitter > 0:
                cmd.extend(["--jitter", f"{self.jitter}ms"])
            if self.packet_loss > 0:
                cmd.extend(["--loss", str(self.packet_loss)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.control_update.emit(f"✅ Network conditions applied to {self.target}\n{result.stdout}")
            else:
                self.control_update.emit(f"⚠️ Network control warning for {self.target}:\n{result.stderr}")
                
        except Exception as e:
            self.control_update.emit(f"❌ Network control error: {str(e)}\n")


class ExperimentRunner(QThread):
    """Background thread for running experiments"""
    progress_update = pyqtSignal(str)
    experiment_finished = pyqtSignal(bool, str)
    
    def __init__(self, command, parent=None):
        super().__init__(parent)
        self.command = command
        self.process = None
        
    def run(self):
        try:
            self.progress_update.emit(f"🚀 Starting experiment...\n")
            self.progress_update.emit(f"Command: {self.command}\n\n")
            
            # Set environment to disable Python buffering for real-time output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=env,    # Force unbuffered Python output
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.progress_update.emit(line)
                if self.process.poll() is not None:
                    break
            
            # Get any remaining output
            remaining = self.process.stdout.read()
            if remaining:
                self.progress_update.emit(remaining)
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.experiment_finished.emit(True, "✅ Experiment completed successfully!")
            else:
                self.experiment_finished.emit(False, f"❌ Experiment failed with code {self.process.returncode}")
                
        except Exception as e:
            self.experiment_finished.emit(False, f"❌ Error: {str(e)}")
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.progress_update.emit("\n⚠️ Experiment stopped by user\n")


class DockerBuildThread(QThread):
    log_update = pyqtSignal(str)
    build_finished = pyqtSignal(bool, int)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd

    def run(self):
        import subprocess
        try:
            proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                if line:
                    self.log_update.emit(line.rstrip())
                if proc.poll() is not None:
                    break
            remaining = proc.stdout.read()
            if remaining:
                self.log_update.emit(remaining)
            proc.wait()
            self.build_finished.emit(True, proc.returncode)
        except Exception as e:
            self.log_update.emit(f"\n❌ Docker build error: {e}")
            self.build_finished.emit(False, -1)


class FLExperimentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.experiment_thread = None
        self.dashboard_thread = None
        self.network_controller = None
        self.log_monitors = []
        self.current_containers = []
        self.experiment_started_at = None
        self.current_results_dir = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🚀 Federated Learning Network Experiment Dashboard")
        
        # Set minimum size and allow window to be resizable/maximizable
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Enable window maximize button and resizing
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
        
        # Control buttons row (above splitter so they never overlap config scroll)
        control_row = self.create_control_buttons_row()
        main_layout.addLayout(control_row)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: Configuration (minimum height so options are visible)
        config_widget = QWidget()
        config_widget.setMinimumHeight(360)
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)
        
        # Tab widget for different configuration sections
        self.config_tabs = QTabWidget()
        self.config_tabs.addTab(self.create_basic_config_tab(), "⚙️ Basic Configuration")
        self.config_tabs.addTab(self.create_network_config_tab(), "🌐 Network Control")
        self.config_tabs.addTab(self.create_advanced_config_tab(), "🔧 Advanced Options")
        self.config_tabs.addTab(self.create_docker_build_tab(), "🐳 Docker Build")
        self.config_tabs.addTab(self.create_docker_cleanup_tab(), "🗑️ Docker Cleanup")
        config_layout.addWidget(self.config_tabs)
        splitter.addWidget(config_widget)
        
        # Bottom section: Monitoring and Output (no buttons here - they are above)
        monitor_output_widget = self.create_monitoring_output_section()
        splitter.addWidget(monitor_output_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready to run experiments")
        
        # Add keyboard shortcuts for window control
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for better UX"""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # F11 for fullscreen toggle
        fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        
        # Ctrl+M for maximize/restore
        maximize_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        maximize_shortcut.activated.connect(self.toggle_maximize)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.statusBar().showMessage("Exited fullscreen mode (F11 to toggle)")
        else:
            self.showFullScreen()
            self.statusBar().showMessage("Fullscreen mode active (F11 to exit)")
    
    def toggle_maximize(self):
        """Toggle maximize/restore window"""
        if self.isMaximized():
            self.showNormal()
            self.statusBar().showMessage("Window restored (Ctrl+M to maximize)")
        else:
            self.showMaximized()
            self.statusBar().showMessage("Window maximized (Ctrl+M to restore)")
        
    def create_header(self):
        """Create header section"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout(header)
        title = QLabel("Federated Learning Network Experiment Dashboard")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Configure and run distributed FL experiments with network simulation")
        subtitle.setStyleSheet("font-size: 14px; color: #f0f0f0;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header
    
    def create_control_buttons_row(self):
        """Create Start/Stop/Apply Network/Clear row - placed above splitter to avoid overlap with config scroll"""
        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)
        
        self.start_button = QPushButton("▶️ Start Experiment")
        self.start_button.setStyleSheet("""
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
        self.start_button.clicked.connect(self.start_experiment)
        
        self.stop_button = QPushButton("⏹️ Stop Experiment")
        self.stop_button.setStyleSheet("""
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
        self.stop_button.clicked.connect(self.stop_experiment)
        self.stop_button.setEnabled(False)
        
        self.apply_network_button = QPushButton("🌐 Apply Network Changes")
        self.apply_network_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #138496; }
        """)
        self.apply_network_button.clicked.connect(self.apply_network_conditions)
        
        self.clear_button = QPushButton("🗑️ Clear All")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        self.clear_button.clicked.connect(self.clear_all_output)
        
        self.diagnostic_pipeline_button = QPushButton("📊 Run Diagnostic Pipeline")
        self.diagnostic_pipeline_button.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover { background-color: #5a32a3; }
            QPushButton:pressed { background-color: #4c2d8a; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.diagnostic_pipeline_button.setToolTip("Run empirical overhead, network extraction, and analytical model for ONE selected protocol (MQTT, AMQP, gRPC, QUIC, or DDS). Single-protocol FL runs separately.")
        self.diagnostic_pipeline_button.clicked.connect(self.start_diagnostic_pipeline)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.diagnostic_pipeline_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.apply_network_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addStretch()
        
        # Progress bar in same row (compact)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setFixedWidth(120)
        control_layout.addWidget(self.progress_bar)
        
        return control_layout
    
    def create_basic_config_tab(self):
        """Create basic configuration tab - vertically scrollable"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(15)
        # Bottom margin so Training Configuration is fully visible when scrolled to end
        layout.setContentsMargins(10, 10, 10, 80)
        
        # Baseline Mode Selection (NEW!)
        baseline_group = QGroupBox("🎯 Experiment Mode")
        baseline_group.setStyleSheet(self.get_group_style())
        baseline_layout = QHBoxLayout()
        
        self.baseline_mode = QCheckBox("Create Baseline Model (Excellent Network, GPU Required)")
        self.baseline_mode.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px; color: #ff6b6b;")
        self.baseline_mode.toggled.connect(self.toggle_baseline_mode)
        baseline_layout.addWidget(self.baseline_mode)
        baseline_layout.addStretch()
        
        baseline_group.setLayout(baseline_layout)
        layout.addWidget(baseline_group)
        
        # Use Case
        use_case_group = self.create_group("📊 Use Case", [
            ("Mental State Recognition", "mentalstate"),
            ("Emotion Recognition", "emotion"),
            ("Temperature Regulation", "temperature")
        ])
        layout.addWidget(use_case_group)
        self.use_case_group = use_case_group
        
        # Protocol Selection
        protocol_group = self.create_checkbox_group("📡 Communication Protocols", [
            ("MQTT", "mqtt", True),
            ("AMQP", "amqp", False),
            ("gRPC", "grpc", False),
            ("QUIC", "quic", False),
            ("HTTP/3", "http3", False),
            ("DDS", "dds", False),
            ("🤖 RL-Unified (Dynamic Selection)", "rl_unified", False)
        ])
        layout.addWidget(protocol_group)
        self.protocol_checkboxes = protocol_group.findChildren(QCheckBox)
        
        # Q-learning convergence (unified use case only)
        ql_conv_group = QGroupBox("🎓 Q-Learning End Condition (Unified Only)")
        ql_conv_group.setStyleSheet(self.get_group_style())
        ql_conv_layout = QHBoxLayout()
        self.use_ql_convergence = QCheckBox("End training when Q-learning value converges (run multiple episodes)")
        self.use_ql_convergence.setStyleSheet("font-size: 12px; padding: 5px;")
        self.use_ql_convergence.setToolTip("If unchecked: training ends on accuracy convergence (current behavior). If checked: training runs until Q-values stabilize.")
        ql_conv_layout.addWidget(self.use_ql_convergence)
        ql_conv_layout.addStretch()
        ql_conv_group.setLayout(ql_conv_layout)
        layout.addWidget(ql_conv_group)
        
        # Network Scenarios
        scenario_group = self.create_checkbox_group("🌐 Network Scenarios", [
            ("Excellent", "excellent", True),
            ("Good", "good", False),
            ("Moderate", "moderate", False),
            ("Poor", "poor", False),
            ("Very Poor", "very_poor", False),
            ("Satellite", "satellite", False),
            ("Light Congestion", "congested_light", False),
            ("Moderate Congestion", "congested_moderate", False),
            ("Heavy Congestion", "congested_heavy", False)
        ])
        layout.addWidget(scenario_group)
        self.scenario_checkboxes = scenario_group.findChildren(QCheckBox)
        
        # GPU Configuration
        gpu_group = QGroupBox("🖥️ GPU Configuration")
        gpu_group.setStyleSheet(self.get_group_style())
        gpu_layout = QHBoxLayout()
        
        self.gpu_enabled = QCheckBox("Enable GPU Acceleration")
        self.gpu_enabled.setChecked(True)
        self.gpu_enabled.setStyleSheet("font-size: 13px; padding: 5px;")
        gpu_layout.addWidget(self.gpu_enabled)
        
        gpu_layout.addWidget(QLabel("GPU Count:"))
        self.gpu_count = QSpinBox()
        self.gpu_count.setRange(0, 8)
        self.gpu_count.setValue(2)
        self.gpu_count.setStyleSheet("padding: 5px;")
        gpu_layout.addWidget(self.gpu_count)
        
        gpu_layout.addStretch()
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # Training Configuration
        training_group = QGroupBox("🎓 Training Configuration")
        training_group.setStyleSheet(self.get_group_style())
        training_layout = QGridLayout()
        
        training_layout.addWidget(QLabel("Number of Rounds:"), 0, 0)
        self.rounds_spinbox = QSpinBox()
        self.rounds_spinbox.setRange(1, 1000)
        self.rounds_spinbox.setValue(10)
        self.rounds_spinbox.setStyleSheet("padding: 5px;")
        training_layout.addWidget(self.rounds_spinbox, 0, 1)
        
        training_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.batch_size.setValue(32)
        self.batch_size.setStyleSheet("padding: 5px;")
        training_layout.addWidget(self.batch_size, 0, 3)
        
        training_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate = QLineEdit("0.001")
        self.learning_rate.setStyleSheet("padding: 5px;")
        training_layout.addWidget(self.learning_rate, 1, 1)
        
        training_layout.addWidget(QLabel("Min Clients:"), 1, 2)
        self.min_clients = QSpinBox()
        self.min_clients.setRange(1, 100)
        self.min_clients.setValue(2)
        self.min_clients.setStyleSheet("padding: 5px;")
        training_layout.addWidget(self.min_clients, 1, 3)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        layout.addStretch()
        
        # Wrap in scroll area so all options are accessible vertically
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMinimumHeight(380)
        return scroll
    
    def create_network_config_tab(self):
        """Create network configuration tab - vertically scrollable"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 80)
        
        # Network Target Selection (NEW!)
        target_group = QGroupBox("🎯 Network Control Target")
        target_group.setStyleSheet(self.get_group_style())
        target_layout = QHBoxLayout()
        
        target_label = QLabel("Apply Network Conditions To:")
        target_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        target_layout.addWidget(target_label)
        
        self.network_target = QComboBox()
        self.network_target.setStyleSheet("padding: 5px; font-size: 12px;")
        self.network_target.addItem("All Clients", "all")
        self.network_target.addItem("Server", "server")
        self.network_target.currentIndexChanged.connect(self.refresh_client_targets)
        target_layout.addWidget(self.network_target)
        
        self.refresh_targets_btn = QPushButton("🔄 Refresh Targets")
        self.refresh_targets_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.refresh_targets_btn.clicked.connect(self.refresh_client_targets)
        target_layout.addWidget(self.refresh_targets_btn)
        
        target_layout.addStretch()
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Dynamic Network Control
        dynamic_group = QGroupBox("🎛️ Dynamic Network Control")
        dynamic_group.setStyleSheet(self.get_group_style())
        dynamic_layout = QGridLayout()
        
        # Latency
        row = 0
        dynamic_layout.addWidget(QLabel("Latency (ms):"), row, 0)
        self.latency_slider = QSlider(Qt.Horizontal)
        self.latency_slider.setRange(0, 1000)
        self.latency_slider.setValue(0)
        self.latency_slider.setTickPosition(QSlider.TicksBelow)
        self.latency_slider.setTickInterval(100)
        self.latency_label = QLabel("0 ms")
        self.latency_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.latency_slider.valueChanged.connect(
            lambda v: self.latency_label.setText(f"{v} ms")
        )
        dynamic_layout.addWidget(self.latency_slider, row, 1)
        dynamic_layout.addWidget(self.latency_label, row, 2)
        
        # Bandwidth
        row += 1
        dynamic_layout.addWidget(QLabel("Bandwidth (Mbps):"), row, 0)
        self.bandwidth_slider = QSlider(Qt.Horizontal)
        self.bandwidth_slider.setRange(1, 1000)
        self.bandwidth_slider.setValue(100)
        self.bandwidth_slider.setTickPosition(QSlider.TicksBelow)
        self.bandwidth_slider.setTickInterval(100)
        self.bandwidth_label = QLabel("100 Mbps")
        self.bandwidth_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.bandwidth_slider.valueChanged.connect(
            lambda v: self.bandwidth_label.setText(f"{v} Mbps")
        )
        dynamic_layout.addWidget(self.bandwidth_slider, row, 1)
        dynamic_layout.addWidget(self.bandwidth_label, row, 2)
        
        # Jitter
        row += 1
        dynamic_layout.addWidget(QLabel("Jitter (ms):"), row, 0)
        self.jitter_slider = QSlider(Qt.Horizontal)
        self.jitter_slider.setRange(0, 100)
        self.jitter_slider.setValue(0)
        self.jitter_slider.setTickPosition(QSlider.TicksBelow)
        self.jitter_slider.setTickInterval(10)
        self.jitter_label = QLabel("0 ms")
        self.jitter_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.jitter_slider.valueChanged.connect(
            lambda v: self.jitter_label.setText(f"{v} ms")
        )
        dynamic_layout.addWidget(self.jitter_slider, row, 1)
        dynamic_layout.addWidget(self.jitter_label, row, 2)
        
        # Packet Loss
        row += 1
        dynamic_layout.addWidget(QLabel("Packet Loss (%):"), row, 0)
        self.packet_loss_slider = QSlider(Qt.Horizontal)
        self.packet_loss_slider.setRange(0, 10)
        self.packet_loss_slider.setValue(0)
        self.packet_loss_slider.setTickPosition(QSlider.TicksBelow)
        self.packet_loss_slider.setTickInterval(1)
        self.packet_loss_label = QLabel("0 %")
        self.packet_loss_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.packet_loss_slider.valueChanged.connect(
            lambda v: self.packet_loss_label.setText(f"{v} %")
        )
        dynamic_layout.addWidget(self.packet_loss_slider, row, 1)
        dynamic_layout.addWidget(self.packet_loss_label, row, 2)
        
        dynamic_group.setLayout(dynamic_layout)
        layout.addWidget(dynamic_group)
        
        # Congestion Control
        congestion_group = QGroupBox("🚦 Traffic Congestion")
        congestion_group.setStyleSheet(self.get_group_style())
        congestion_layout = QVBoxLayout()
        
        self.enable_congestion = QCheckBox("Enable Traffic Generator-Based Congestion")
        self.enable_congestion.setStyleSheet("font-size: 13px; padding: 5px;")
        congestion_layout.addWidget(self.enable_congestion)
        
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Congestion Level:"))
        self.congestion_level = QComboBox()
        self.congestion_level.addItems(["Light", "Moderate", "Heavy"])
        self.congestion_level.setCurrentText("Moderate")
        self.congestion_level.setStyleSheet("padding: 5px;")
        level_layout.addWidget(self.congestion_level)
        level_layout.addStretch()
        congestion_layout.addLayout(level_layout)
        
        congestion_group.setLayout(congestion_layout)
        layout.addWidget(congestion_group)
        
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMinimumHeight(360)
        return scroll
    
    def create_advanced_config_tab(self):
        """Create advanced configuration tab - vertically scrollable"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 80)
        
        # Quantization
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
        self.quant_bits.addItems(["8", "16", "32"])
        self.quant_bits.setCurrentText("8")
        self.quant_bits.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_bits, 0, 1)
        
        quant_options_layout.addWidget(QLabel("Strategy:"), 0, 2)
        self.quant_strategy = QComboBox()
        self.quant_strategy.addItems([
            "full_quantization",
            "parameter_quantization",
            "activation_quantization"
        ])
        self.quant_strategy.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_strategy, 0, 3)
        
        self.quant_symmetric = QCheckBox("Use Symmetric Quantization")
        self.quant_symmetric.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_symmetric, 1, 0, 1, 2)
        
        self.quant_per_channel = QCheckBox("Per-Channel Quantization")
        self.quant_per_channel.setStyleSheet("padding: 5px;")
        quant_options_layout.addWidget(self.quant_per_channel, 1, 2, 1, 2)
        
        quant_options.setLayout(quant_options_layout)
        quant_layout.addWidget(quant_options)
        self.quant_options_widget = quant_options
        quant_options.setEnabled(False)
        
        quant_group.setLayout(quant_layout)
        layout.addWidget(quant_group)
        
        # Compression
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
        compression_layout.addWidget(comp_options)
        self.comp_options_widget = comp_options
        comp_options.setEnabled(False)
        
        compression_group.setLayout(compression_layout)
        layout.addWidget(compression_group)
        
        # Pruning
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
        self.pruning_ratio.valueChanged.connect(
            lambda v: self.pruning_ratio_label.setText(f"{v}%")
        )
        pruning_options_layout.addWidget(self.pruning_ratio)
        pruning_options_layout.addWidget(self.pruning_ratio_label)
        
        pruning_options.setLayout(pruning_options_layout)
        pruning_layout.addWidget(pruning_options)
        self.pruning_options_widget = pruning_options
        pruning_options.setEnabled(False)
        
        pruning_group.setLayout(pruning_layout)
        layout.addWidget(pruning_group)
        
        # Other Options
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
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMinimumHeight(360)
        return scroll
    
    def create_docker_build_tab(self):
        """Create Docker build tab with build button - vertically scrollable"""
        import subprocess
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 80)

        docker_group = QGroupBox("🐳 Docker Image Build Options")
        docker_group.setStyleSheet(self.get_group_style())
        docker_layout = QVBoxLayout()

        self.rebuild_images = QCheckBox("Rebuild Docker Images Before Experiment")
        self.rebuild_images.setStyleSheet("font-size: 13px; padding: 5px;")
        docker_layout.addWidget(self.rebuild_images)

        self.use_cache = QCheckBox("Use Cache When Building Docker Images")
        self.use_cache.setChecked(True)
        self.use_cache.setStyleSheet("font-size: 13px; padding: 5px;")
        docker_layout.addWidget(self.use_cache)

        # Add Build Docker Images button
        self.build_btn_temp = QPushButton("🐳 Build Docker Images (Temperature Regulation)")
        self.build_btn_temp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #3498db; color: white; border-radius: 6px;")
        self.build_btn_temp.clicked.connect(self.build_docker_images_temperature)
        docker_layout.addWidget(self.build_btn_temp)

        # Add Build Docker Images button
        self.build_btn_mental = QPushButton("🐳 Build Docker Images (Mental State)")
        self.build_btn_mental.setStyleSheet("font-size: 15px; padding: 10px; background-color: #3498db; color: white; border-radius: 6px;")
        self.build_btn_mental.clicked.connect(self.build_docker_images_mentalstate)
        docker_layout.addWidget(self.build_btn_mental)

        # Add Build Docker Images button
        self.build_btn_emotion = QPushButton("🐳 Build Docker Images (Emotion)")
        self.build_btn_emotion.setStyleSheet("font-size: 15px; padding: 10px; background-color: #3498db; color: white; border-radius: 6px;")
        self.build_btn_emotion.clicked.connect(self.build_docker_images_emotion)
        docker_layout.addWidget(self.build_btn_emotion)

        # Add separator
        separator = QLabel("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        separator.setStyleSheet("color: #95a5a6; font-size: 10px; margin: 10px 0;")
        docker_layout.addWidget(separator)
        
        unified_label = QLabel("🤖 RL-Unified Scenario (Dynamic Protocol Selection)")
        unified_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e74c3c; margin-top: 10px;")
        docker_layout.addWidget(unified_label)

        # Add Build Docker Images button for Unified scenarios
        self.build_btn_unified_temp = QPushButton("🤖 Build Unified Images (Temperature Regulation)")
        self.build_btn_unified_temp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #e74c3c; color: white; border-radius: 6px;")
        self.build_btn_unified_temp.clicked.connect(self.build_docker_images_unified_temperature)
        docker_layout.addWidget(self.build_btn_unified_temp)

        self.build_btn_unified_mental = QPushButton("🤖 Build Unified Images (Mental State)")
        self.build_btn_unified_mental.setStyleSheet("font-size: 15px; padding: 10px; background-color: #e74c3c; color: white; border-radius: 6px;")
        self.build_btn_unified_mental.clicked.connect(self.build_docker_images_unified_mentalstate)
        docker_layout.addWidget(self.build_btn_unified_mental)

        self.build_btn_unified_emotion = QPushButton("🤖 Build Unified Images (Emotion)")
        self.build_btn_unified_emotion.setStyleSheet("font-size: 15px; padding: 10px; background-color: #e74c3c; color: white; border-radius: 6px;")
        self.build_btn_unified_emotion.clicked.connect(self.build_docker_images_unified_emotion)
        docker_layout.addWidget(self.build_btn_unified_emotion)

        docker_group.setLayout(docker_layout)
        layout.addWidget(docker_group)

        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMinimumHeight(360)
        return scroll
    
    def create_docker_cleanup_tab(self):
        """Create Docker cleanup tab for removing images - vertically scrollable"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 80)
        
        # Single Protocol Images Cleanup
        single_group = QGroupBox("🗑️ Delete Single Protocol Images")
        single_group.setStyleSheet(self.get_group_style())
        single_layout = QVBoxLayout()
        
        info_label = QLabel("Delete images for specific use case (all protocols)")
        info_label.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        single_layout.addWidget(info_label)
        
        self.cleanup_btn_emotion = QPushButton("🗑️ Delete All Emotion Images (Single Protocol)")
        self.cleanup_btn_emotion.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e67e22; color: white; border-radius: 6px;")
        self.cleanup_btn_emotion.clicked.connect(lambda: self.cleanup_docker_images("emotion", False))
        single_layout.addWidget(self.cleanup_btn_emotion)
        
        self.cleanup_btn_mental = QPushButton("🗑️ Delete All Mental State Images (Single Protocol)")
        self.cleanup_btn_mental.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e67e22; color: white; border-radius: 6px;")
        self.cleanup_btn_mental.clicked.connect(lambda: self.cleanup_docker_images("mentalstate", False))
        single_layout.addWidget(self.cleanup_btn_mental)
        
        self.cleanup_btn_temp = QPushButton("🗑️ Delete All Temperature Images (Single Protocol)")
        self.cleanup_btn_temp.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e67e22; color: white; border-radius: 6px;")
        self.cleanup_btn_temp.clicked.connect(lambda: self.cleanup_docker_images("temperature", False))
        single_layout.addWidget(self.cleanup_btn_temp)
        
        single_group.setLayout(single_layout)
        layout.addWidget(single_group)
        
        # Unified Images Cleanup
        unified_group = QGroupBox("🤖 Delete Unified (RL) Images")
        unified_group.setStyleSheet(self.get_group_style())
        unified_layout = QVBoxLayout()
        
        info_label2 = QLabel("Delete unified scenario images for specific use case")
        info_label2.setStyleSheet("font-size: 12px; color: #666; padding: 5px;")
        unified_layout.addWidget(info_label2)
        
        self.cleanup_btn_unified_emotion = QPushButton("🤖 Delete Unified Emotion Images")
        self.cleanup_btn_unified_emotion.setStyleSheet("font-size: 14px; padding: 10px; background-color: #c0392b; color: white; border-radius: 6px;")
        self.cleanup_btn_unified_emotion.clicked.connect(lambda: self.cleanup_docker_images("emotion", True))
        unified_layout.addWidget(self.cleanup_btn_unified_emotion)
        
        self.cleanup_btn_unified_mental = QPushButton("🤖 Delete Unified Mental State Images")
        self.cleanup_btn_unified_mental.setStyleSheet("font-size: 14px; padding: 10px; background-color: #c0392b; color: white; border-radius: 6px;")
        self.cleanup_btn_unified_mental.clicked.connect(lambda: self.cleanup_docker_images("mentalstate", True))
        unified_layout.addWidget(self.cleanup_btn_unified_mental)
        
        self.cleanup_btn_unified_temp = QPushButton("🤖 Delete Unified Temperature Images")
        self.cleanup_btn_unified_temp.setStyleSheet("font-size: 14px; padding: 10px; background-color: #c0392b; color: white; border-radius: 6px;")
        self.cleanup_btn_unified_temp.clicked.connect(lambda: self.cleanup_docker_images("temperature", True))
        unified_layout.addWidget(self.cleanup_btn_unified_temp)
        
        unified_group.setLayout(unified_layout)
        layout.addWidget(unified_group)
        
        # All Images Cleanup
        all_group = QGroupBox("💣 Delete ALL Docker Images")
        all_group.setStyleSheet(self.get_group_style())
        all_layout = QVBoxLayout()
        
        warning_label = QLabel("⚠️ WARNING: This will delete ALL FL Docker images!")
        warning_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #c0392b; padding: 5px;")
        all_layout.addWidget(warning_label)
        
        self.cleanup_btn_all = QPushButton("💣 Delete ALL FL Docker Images")
        self.cleanup_btn_all.setStyleSheet("font-size: 14px; padding: 10px; background-color: #8b0000; color: white; border-radius: 6px;")
        self.cleanup_btn_all.clicked.connect(self.cleanup_all_docker_images)
        all_layout.addWidget(self.cleanup_btn_all)
        
        all_group.setLayout(all_layout)
        layout.addWidget(all_group)
        
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setMinimumHeight(360)
        return scroll
    
    def cleanup_docker_images(self, use_case, unified=False):
        """Delete Docker images for specific use case"""
        import subprocess
        
        # Build image name patterns
        if unified:
            pattern = f"docker-fl-.*-unified-{use_case}"
            desc = f"unified {use_case}"
        else:
            # For single protocol, match all protocol-specific images
            pattern = f"docker-fl-.*(mqtt|amqp|grpc|quic|dds)-{use_case}"
            desc = f"single protocol {use_case}"
        
        # List matching images
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                QMessageBox.critical(self, "Error", "Failed to list Docker images")
                return
            
            import re
            images = [img for img in result.stdout.strip().split('\n') 
                     if img and re.search(pattern, img)]
            
            if not images:
                QMessageBox.information(self, "No Images Found", 
                                       f"No {desc} images found to delete.")
                return
            
            # Confirm deletion
            image_list = '\n'.join(images)
            reply = QMessageBox.question(
                self, "Confirm Deletion",
                f"Found {len(images)} image(s) to delete:\n\n{image_list}\n\n"
                f"This will free up approximately {len(images) * 7.5:.1f} GB.\n\n"
                "Are you sure you want to delete these images?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Disable appropriate button
            if unified:
                if use_case == "emotion":
                    self.cleanup_btn_unified_emotion.setEnabled(False)
                elif use_case == "mentalstate":
                    self.cleanup_btn_unified_mental.setEnabled(False)
                else:
                    self.cleanup_btn_unified_temp.setEnabled(False)
            else:
                if use_case == "emotion":
                    self.cleanup_btn_emotion.setEnabled(False)
                elif use_case == "mentalstate":
                    self.cleanup_btn_mental.setEnabled(False)
                else:
                    self.cleanup_btn_temp.setEnabled(False)
            
            # Delete images
            self.docker_build_log_text.clear()
            self.output_tabs.setCurrentWidget(self.docker_build_log_text)
            self.docker_build_log_text.append(f"🗑️ Deleting {len(images)} {desc} images...\\n")
            self.statusBar().showMessage(f"Deleting {len(images)} {desc} images...")
            
            deleted = 0
            failed = 0
            for img in images:
                try:
                    self.docker_build_log_text.append(f"Deleting: {img}")
                    result = subprocess.run(
                        ["docker", "rmi", "-f", img],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        self.docker_build_log_text.append(f"  ✅ Deleted successfully")
                        deleted += 1
                    else:
                        self.docker_build_log_text.append(f"  ❌ Failed: {result.stderr}")
                        failed += 1
                except Exception as e:
                    self.docker_build_log_text.append(f"  ❌ Error: {str(e)}")
                    failed += 1
            
            # Re-enable button
            if unified:
                if use_case == "emotion":
                    self.cleanup_btn_unified_emotion.setEnabled(True)
                elif use_case == "mentalstate":
                    self.cleanup_btn_unified_mental.setEnabled(True)
                else:
                    self.cleanup_btn_unified_temp.setEnabled(True)
            else:
                if use_case == "emotion":
                    self.cleanup_btn_emotion.setEnabled(True)
                elif use_case == "mentalstate":
                    self.cleanup_btn_mental.setEnabled(True)
                else:
                    self.cleanup_btn_temp.setEnabled(True)
            
            # Show summary
            self.docker_build_log_text.append(f"\\n📊 Summary: {deleted} deleted, {failed} failed")
            self.statusBar().showMessage(f"✅ Cleanup complete: {deleted} deleted, {failed} failed")
            QMessageBox.information(self, "Cleanup Complete",
                                   f"Deleted {deleted} images\\nFailed: {failed}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during cleanup: {str(e)}")
            self.statusBar().showMessage("❌ Cleanup failed")
    
    def cleanup_all_docker_images(self):
        """Delete ALL FL Docker images"""
        import subprocess
        
        # List all FL images
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                QMessageBox.critical(self, "Error", "Failed to list Docker images")
                return
            
            images = [img for img in result.stdout.strip().split('\\n') 
                     if img and 'docker-fl-' in img]
            
            if not images:
                QMessageBox.information(self, "No Images Found", 
                                       "No FL Docker images found to delete.")
                return
            
            # Strong confirmation
            reply = QMessageBox.warning(
                self, "⚠️ CONFIRM DELETION OF ALL IMAGES",
                f"⚠️ WARNING: You are about to delete ALL {len(images)} FL Docker images!\\n\\n"
                f"This will free up approximately {len(images) * 7.5:.1f} GB.\\n\\n"
                "This action cannot be undone!\\n\\n"
                "Are you ABSOLUTELY SURE?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Disable all cleanup buttons
            self.cleanup_btn_all.setEnabled(False)
            
            # Delete images
            self.docker_build_log_text.clear()
            self.output_tabs.setCurrentWidget(self.docker_build_log_text)
            self.docker_build_log_text.append(f"💣 Deleting ALL {len(images)} FL Docker images...\\n")
            self.statusBar().showMessage(f"Deleting all {len(images)} FL images...")
            
            deleted = 0
            failed = 0
            for img in images:
                try:
                    self.docker_build_log_text.append(f"Deleting: {img}")
                    result = subprocess.run(
                        ["docker", "rmi", "-f", img],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        self.docker_build_log_text.append(f"  ✅ Deleted")
                        deleted += 1
                    else:
                        self.docker_build_log_text.append(f"  ❌ Failed")
                        failed += 1
                except Exception as e:
                    self.docker_build_log_text.append(f"  ❌ Error: {str(e)}")
                    failed += 1
            
            # Re-enable button
            self.cleanup_btn_all.setEnabled(True)
            
            # Show summary
            self.docker_build_log_text.append(f"\\n📊 Final Summary: {deleted} deleted, {failed} failed")
            self.statusBar().showMessage(f"✅ Cleanup complete: {deleted} deleted, {failed} failed")
            QMessageBox.information(self, "Cleanup Complete",
                                   f"Deleted {deleted} images\\nFailed: {failed}\\n\\n"
                                   f"Freed approximately {deleted * 7.5:.1f} GB")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during cleanup: {str(e)}")
            self.statusBar().showMessage("❌ Cleanup failed")
            self.cleanup_btn_all.setEnabled(True)

    def build_docker_images_emotion(self):
        """Build Docker images for the emotion use case and show output"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-emotion.gpu-isolated.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🐳 Building Docker images for Emotion use case...")
        
        # Disable the button
        self.build_btn_emotion.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        
        # Start Docker build in a thread
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            self.build_btn_emotion.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Docker images built successfully (Emotion)")
                self.docker_build_log_text.append("\n✅ Docker images built successfully (Emotion)")
            else:
                self.statusBar().showMessage("❌ Docker build failed (Emotion)")
                self.docker_build_log_text.append(f"\n❌ Docker build failed (Emotion), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()
    
    def build_docker_images_mentalstate(self):
        """Build Docker images for the mental state use case and show output"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-mentalstate.gpu-isolated.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🐳 Building Docker images for Mental State use case...")
        
        # Disable the button
        self.build_btn_mental.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        
        # Start Docker build in a thread
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            self.build_btn_mental.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Docker images built successfully (Mental State)")
                self.docker_build_log_text.append("\n✅ Docker images built successfully (Mental State)")
            else:
                self.statusBar().showMessage("❌ Docker build failed (Mental State)")
                self.docker_build_log_text.append(f"\n❌ Docker build failed (Mental State), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()

    def build_docker_images_temperature(self):
        """Build Docker images for the temperature regulation use case and show output"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-temperature.gpu-isolated.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🐳 Building Docker images for Temperature Regulation use case...")
        
        # Disable the button
        self.build_btn_temp.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        
        # Start Docker build in a thread
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            self.build_btn_temp.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Docker images built successfully (Temperature Regulation)")
                self.docker_build_log_text.append("\n✅ Docker images built successfully (Temperature Regulation)")
            else:
                self.statusBar().showMessage("❌ Docker build failed (Temperature Regulation)")
                self.docker_build_log_text.append(f"\n❌ Docker build failed (Temperature Regulation), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()

    def build_docker_images_unified_temperature(self):
        """Build Docker images for unified RL scenario - temperature use case"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-unified-temperature.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🤖 Building Unified Docker images for Temperature Regulation...")
        if hasattr(self, 'build_btn_unified_temp'):
            self.build_btn_unified_temp.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            if hasattr(self, 'build_btn_unified_temp'):
                self.build_btn_unified_temp.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Unified images built successfully (Temperature)")
                self.docker_build_log_text.append("\n✅ Unified Docker images built successfully (Temperature)")
            else:
                self.statusBar().showMessage("❌ Unified build failed (Temperature)")
                self.docker_build_log_text.append(f"\n❌ Unified Docker build failed (Temperature), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()

    def build_docker_images_unified_mentalstate(self):
        """Build Docker images for unified RL scenario - mentalstate use case"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-unified-mentalstate.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🤖 Building Unified Docker images for Mental State...")
        if hasattr(self, 'build_btn_unified_mental'):
            self.build_btn_unified_mental.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            if hasattr(self, 'build_btn_unified_mental'):
                self.build_btn_unified_mental.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Unified images built successfully (Mental State)")
                self.docker_build_log_text.append("\n✅ Unified Docker images built successfully (Mental State)")
            else:
                self.statusBar().showMessage("❌ Unified build failed (Mental State)")
                self.docker_build_log_text.append(f"\n❌ Unified Docker build failed (Mental State), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()

    def build_docker_images_unified_emotion(self):
        """Build Docker images for unified RL scenario - emotion use case"""
        use_cache = self.use_cache.isChecked()
        compose_file = "Docker/docker-compose-unified-emotion.yml"
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        full_compose_path = f"{base_dir}/{compose_file}"
        cmd = ["docker", "compose", "-f", full_compose_path, "build"]
        if not use_cache:
            cmd.append("--no-cache")
        self.statusBar().showMessage("🤖 Building Unified Docker images for Emotion...")
        if hasattr(self, 'build_btn_unified_emotion'):
            self.build_btn_unified_emotion.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            if hasattr(self, 'build_btn_unified_emotion'):
                self.build_btn_unified_emotion.setEnabled(True)
            if success and returncode == 0:
                self.statusBar().showMessage("✅ Unified images built successfully (Emotion)")
                self.docker_build_log_text.append("\n✅ Unified Docker images built successfully (Emotion)")
            else:
                self.statusBar().showMessage("❌ Unified build failed (Emotion)")
                self.docker_build_log_text.append(f"\n❌ Unified Docker build failed (Emotion), code {returncode}")
        self.docker_build_thread.build_finished.connect(on_build_finished)
        self.docker_build_thread.start()

    def create_monitoring_output_section(self):
        """Create comprehensive monitoring and output section (buttons are in create_control_buttons_row above)"""
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Tabbed output section (Start/Stop/Apply Network/Clear are above the splitter)
        self.output_tabs = QTabWidget()
        
        # Tab 1: Experiment Output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
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
        self.output_tabs.addTab(self.output_text, "📊 Experiment Output")
        
        # Tab 2: Training Dashboard with Baseline Comparison
        self.dashboard_text = QTextEdit()
        self.dashboard_text.setReadOnly(True)
        self.dashboard_text.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #58a6ff;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.output_tabs.addTab(self.dashboard_text, "📈 FL Training Monitor (vs Baseline)")
        
        # Tab 3: Server Logs
        server_tab_widget = QWidget()
        server_tab_layout = QVBoxLayout(server_tab_widget)
        server_tab_layout.setSpacing(5)
        server_toolbar = QHBoxLayout()
        server_toolbar.addWidget(QLabel("Select Server:"))
        self.server_selector = QComboBox()
        self.server_selector.setStyleSheet("padding: 5px; font-size: 11px; min-width: 150px;")
        self.server_selector.addItem("Detecting servers...", None)
        self.server_selector.currentIndexChanged.connect(self.switch_server_log)
        server_toolbar.addWidget(self.server_selector)
        self.refresh_servers_btn = QPushButton("🔄 Refresh")
        self.refresh_servers_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.refresh_servers_btn.clicked.connect(self.refresh_server_list)
        server_toolbar.addWidget(self.refresh_servers_btn)
        server_toolbar.addStretch()
        server_tab_layout.addLayout(server_toolbar)
        self.server_log_text = QTextEdit()
        self.server_log_text.setReadOnly(True)
        self.server_log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a2e;
                color: #16c79a;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        server_tab_layout.addWidget(self.server_log_text)
        self.output_tabs.addTab(server_tab_widget, "🖥️ Server Logs")

        # Tab 4: Client Logs
        client_tab_widget = QWidget()
        client_tab_layout = QVBoxLayout(client_tab_widget)
        client_tab_layout.setSpacing(5)
        client_toolbar = QHBoxLayout()
        client_toolbar.addWidget(QLabel("Select Client:"))
        self.client_selector = QComboBox()
        self.client_selector.setStyleSheet("padding: 5px; font-size: 11px; min-width: 150px;")
        self.client_selector.addItem("Detecting clients...", None)
        self.client_selector.currentIndexChanged.connect(self.switch_client_log)
        client_toolbar.addWidget(self.client_selector)
        self.refresh_clients_btn = QPushButton("🔄 Refresh")
        self.refresh_clients_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.refresh_clients_btn.clicked.connect(self.refresh_client_list)
        client_toolbar.addWidget(self.refresh_clients_btn)
        client_toolbar.addStretch()
        client_tab_layout.addLayout(client_toolbar)
        self.client_log_text = QTextEdit()
        self.client_log_text.setReadOnly(True)
        self.client_log_text.setStyleSheet("""
            QTextEdit {
                background-color: #16213e;
                color: #f9a826;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        client_tab_layout.addWidget(self.client_log_text)
        self.output_tabs.addTab(client_tab_widget, "💻 Client Logs")

        # Tab 5: Packet Logs (Full-Featured Component)
        try:
            from packet_logs_tab import PacketLogsTab
            self.packet_logs_tab = PacketLogsTab()
            self.output_tabs.addTab(self.packet_logs_tab, "📦 Packet Logs")
        except ImportError as e:
            # Fallback to simple table if PacketLogsTab not available
            packet_tab_widget = QWidget()
            packet_tab_layout = QVBoxLayout(packet_tab_widget)
            packet_tab_layout.setSpacing(5)
            error_label = QLabel(f"⚠️ Advanced Packet Logs unavailable: {e}")
            error_label.setStyleSheet("color: orange; padding: 10px;")
            packet_tab_layout.addWidget(error_label)
            refresh_btn = QPushButton("🔄 Refresh Packet Logs")
            refresh_btn.setStyleSheet("padding: 6px 12px; font-size: 12px;")
            refresh_btn.clicked.connect(self.refresh_packet_log_table)
            packet_tab_layout.addWidget(refresh_btn)
            self.packet_log_table = QTableWidget()
            self.packet_log_table.setColumnCount(7)
            self.packet_log_table.setHorizontalHeaderLabels([
                "Type", "Timestamp", "Size (B)", "Peer", "Protocol", "Round", "Extra Info"
            ])
            self.packet_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.packet_log_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.packet_log_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.packet_log_table.setStyleSheet("font-size: 11px;")
            packet_tab_layout.addWidget(self.packet_log_table)
            packet_tab_widget.setLayout(packet_tab_layout)
            self.output_tabs.addTab(packet_tab_widget, "📦 Packet Logs (Simple)")
            QTimer.singleShot(100, self.refresh_packet_log_table)
            self.packet_logs_tab = None

        # Tab 6: Q-Learning Logs (unified use case)
        try:
            from q_learning_logs_tab import QLearningLogsTab
            self.q_learning_logs_tab = QLearningLogsTab()
            self.output_tabs.addTab(self.q_learning_logs_tab, "🎓 Q-Learning")
        except ImportError as e:
            ql_fallback = QWidget()
            ql_layout = QVBoxLayout(ql_fallback)
            ql_layout.addWidget(QLabel(f"Q-Learning tab unavailable: {e}"))
            self.output_tabs.addTab(ql_fallback, "🎓 Q-Learning")
            self.q_learning_logs_tab = None

        # Tab: Diagnostic Pipeline Results (beside Q-Learning; filled when Run Diagnostic Pipeline completes)
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        diag_tab = QWidget()
        diag_layout = QVBoxLayout(diag_tab)
        diag_layout.addWidget(QLabel("Diagnostic pipeline results (Empirical Overhead, Network Extraction, Analytical Model). Only \"Run Diagnostic Pipeline\" fills this tab — \"Start Experiment\" does not."))
        self.diagnostic_results_table = QTableWidget()
        self.diagnostic_results_table.setColumnCount(9)
        self.diagnostic_results_table.setHorizontalHeaderLabels([
            "Client", "Protocol", "Scenario", "O_app (s)", "O_broker", "Loss (p)", "T_actual (s)", "T_calc (s)", "Error %"
        ])
        self.diagnostic_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.diagnostic_results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.diagnostic_results_table.setStyleSheet("""
            QTableWidget {
                background-color: #222831;
                color: #eee;
                gridline-color: #444;
                font-size: 12px;
            }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { background-color: #393e46; color: #00adb5; padding: 8px; }
        """)
        diag_layout.addWidget(self.diagnostic_results_table)
        self.output_tabs.addTab(diag_tab, "📊 Diagnostic Results")

        # Add Docker Build Logs tab
        self.docker_build_log_text = QTextEdit()
        self.docker_build_log_text.setReadOnly(True)
        self.docker_build_log_text.setStyleSheet("""
            QTextEdit {
                background-color: #222831;
                color: #00adb5;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        self.output_tabs.addTab(self.docker_build_log_text, "🐳 Docker Build Logs")

        layout.addWidget(self.output_tabs)
        return widget

    def refresh_packet_log_table(self):
        """Refresh packet log table (fallback simple version)"""
        import sqlite3
        # Check shared_data directory first (for unified scenario)
        shared_db_server = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared_data', 'packet_logs_server.db')
        # Fallback to old location
        old_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'packet_logs.db')
        
        db_path = shared_db_server if os.path.exists(shared_db_server) else old_db_path
        
        if not hasattr(self, 'packet_log_table'):
            return  # Skip if using PacketLogsTab component
        
        self.packet_log_table.setRowCount(0)
        try:
            if not os.path.exists(db_path):
                self.packet_log_table.setRowCount(1)
                from PyQt5.QtWidgets import QTableWidgetItem
                self.packet_log_table.setItem(0, 0, QTableWidgetItem("Info"))
                self.packet_log_table.setItem(0, 1, QTableWidgetItem(f"No database found at {db_path}"))
                return
            
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('SELECT timestamp, packet_size, peer, protocol, round, extra_info FROM sent_packets ORDER BY timestamp DESC LIMIT 100')
            sent_rows = c.fetchall()
            c.execute('SELECT timestamp, packet_size, peer, protocol, round, extra_info FROM received_packets ORDER BY timestamp DESC LIMIT 100')
            recv_rows = c.fetchall()
            conn.close()
            all_rows = [("Sent",) + row for row in sent_rows] + [("Received",) + row for row in recv_rows]
            all_rows.sort(key=lambda r: r[1], reverse=True)
            self.packet_log_table.setRowCount(len(all_rows))
            for i, row in enumerate(all_rows):
                for j, val in enumerate(row):
                    from PyQt5.QtWidgets import QTableWidgetItem
                    item = QTableWidgetItem(str(val) if val is not None else "")
                    self.packet_log_table.setItem(i, j, item)
        except Exception as e:
            self.packet_log_table.setRowCount(1)
            from PyQt5.QtWidgets import QTableWidgetItem
            self.packet_log_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.packet_log_table.setItem(0, 1, QTableWidgetItem(str(e)))
    
    def create_group(self, title, options):
        """Create a radio button group"""
        group = QGroupBox(title)
        group.setStyleSheet(self.get_group_style())
        layout = QHBoxLayout()
        
        button_group = QButtonGroup(group)
        for i, (label, value) in enumerate(options):
            radio = QRadioButton(label)
            radio.setProperty("value", value)
            radio.setStyleSheet("font-size: 13px; padding: 5px;")
            if i == 0:
                radio.setChecked(True)
            button_group.addButton(radio)
            layout.addWidget(radio)
        
        layout.addStretch()
        group.setLayout(layout)
        group.button_group = button_group
        return group
    
    def create_checkbox_group(self, title, options):
        """Create a checkbox group with optional scrolling for many options"""
        group = QGroupBox(title)
        group.setStyleSheet(self.get_group_style())
        
        # Determine if we need scroll area (more than 6 options)
        if len(options) > 6:
            # Create scroll area for many options
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setMaximumHeight(250)
            scroll.setStyleSheet("QScrollArea { background: transparent; }")
            
            content = QWidget()
            layout = QGridLayout(content)
            layout.setSpacing(8)
            
            for i, (label, value, checked) in enumerate(options):
                checkbox = QCheckBox(label)
                checkbox.setProperty("value", value)
                checkbox.setChecked(checked)
                checkbox.setStyleSheet("font-size: 13px; padding: 5px;")
                row = i // 3
                col = i % 3
                layout.addWidget(checkbox, row, col)
            
            scroll.setWidget(content)
            
            main_layout = QVBoxLayout()
            main_layout.addWidget(scroll)
            group.setLayout(main_layout)
        else:
            # Simple grid layout for few options
            layout = QGridLayout()
            layout.setSpacing(8)
            
            for i, (label, value, checked) in enumerate(options):
                checkbox = QCheckBox(label)
                checkbox.setProperty("value", value)
                checkbox.setChecked(checked)
                checkbox.setStyleSheet("font-size: 13px; padding: 5px;")
                row = i // 3
                col = i % 3
                layout.addWidget(checkbox, row, col)
            
            group.setLayout(layout)
        
        return group
    
    def toggle_quantization_options(self, enabled):
        """Toggle quantization options"""
        self.quant_options_widget.setEnabled(enabled)
    
    def toggle_compression_options(self, enabled):
        """Toggle compression options"""
        self.comp_options_widget.setEnabled(enabled)
    
    def toggle_pruning_options(self, enabled):
        """Toggle pruning options"""
        self.pruning_options_widget.setEnabled(enabled)
    
    def toggle_baseline_mode(self, enabled):
        """Toggle baseline mode - disables network controls and forces GPU"""
        if enabled:
            # Force GPU when baseline mode is enabled
            self.gpu_enabled.setChecked(True)
            self.gpu_enabled.setEnabled(False)
            
            # Disable network controls tab
            self.config_tabs.setTabEnabled(1, False)  # Network Control tab is index 1
            
            # Show warning
            self.output_text.append(
                "⚠️ BASELINE MODE ENABLED:\n"
                "  • GPU: FORCED ON (required for baseline)\n"
                "  • Network Controls: DISABLED (excellent network conditions)\n"
                "  • Baseline models will be stored in experiment_results_baseline/\n"
            )
        else:
            # Re-enable GPU checkbox
            self.gpu_enabled.setEnabled(True)
            
            # Re-enable network controls tab
            self.config_tabs.setTabEnabled(1, True)
            
            self.output_text.append("✅ Baseline mode disabled - network controls available\n")
    
    def refresh_client_targets(self):
        """Refresh available client targets for network control"""
        try:
            # Get running client containers
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=client', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                clients = [c.strip() for c in result.stdout.strip().split('\n') if c.strip()]
                
                # Update network target dropdown - block signals to prevent recursion
                current_value = self.network_target.currentData()
                
                # Block signals temporarily to prevent infinite recursion
                self.network_target.blockSignals(True)
                
                try:
                    self.network_target.clear()
                    self.network_target.addItem("All Clients", "all")
                    self.network_target.addItem("Server", "server")
                    
                    for client in clients:
                        # Extract client number from container name
                        if 'client' in client.lower():
                            client_label = client.replace('_', ' ').title()
                            self.network_target.addItem(f"🖥️ {client_label}", client)
                    
                    # Restore previous selection if still available
                    index = self.network_target.findData(current_value)
                    if index >= 0:
                        self.network_target.setCurrentIndex(index)
                finally:
                    # Re-enable signals
                    self.network_target.blockSignals(False)
                
                self.output_text.append(f"✅ Detected {len(clients)} client container(s)\n")
            
        except Exception as e:
            self.output_text.append(f"⚠️ Could not detect client targets: {e}\n")
    
    def refresh_server_list(self):
        """Refresh the list of available server containers"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=server', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                servers = [c.strip() for c in result.stdout.strip().split('\n') if c.strip()]
                
                # Filter to only FL servers (exclude brokers)
                fl_servers = [s for s in servers if 'fl-' in s.lower() and 'server' in s.lower()]
                
                # Update server selector dropdown
                current_server = self.server_selector.currentData()
                self.server_selector.clear()
                
                if fl_servers:
                    for server in fl_servers:
                        # Extract protocol and use case from container name
                        server_label = server.replace('fl-server-', '').replace('-', ' ').title()
                        self.server_selector.addItem(f"🖥️ {server_label}", server)
                    
                    self.output_text.append(f"✅ Detected {len(fl_servers)} server(s) for log viewing\n")
                    
                    # If no server was selected or current server not in list, select first one
                    if current_server not in fl_servers:
                        self.switch_server_log()
                else:
                    self.server_selector.addItem("No servers detected", None)
                    self.output_text.append("⚠️ No server containers detected\n")
            
        except Exception as e:
            self.output_text.append(f"⚠️ Could not detect servers: {e}\n")
            self.server_selector.clear()
            self.server_selector.addItem("Error detecting servers", None)
    
    def switch_server_log(self):
        """Switch to viewing logs of a different server"""
        server_name = self.server_selector.currentData()
        
        if server_name is None:
            return
        
        # Stop current server log monitor if running
        server_monitor = None
        for monitor in self.log_monitors:
            if monitor.log_type == "server" and monitor.isRunning():
                server_monitor = monitor
                break
        
        if server_monitor:
            server_monitor.stop()
            server_monitor.wait()
            if server_monitor in self.log_monitors:
                self.log_monitors.remove(server_monitor)
        
        # Clear current log display
        self.server_log_text.clear()
        self.server_log_text.append(f"=== Logs for {server_name} ===\n")
        
        # Start new log monitor for selected server
        new_server_monitor = LogMonitor(server_name, "server")
        new_server_monitor.log_update.connect(
            lambda log_type, text: self.server_log_text.append(text)
        )
        new_server_monitor.start()
        self.log_monitors.append(new_server_monitor)
    
    def refresh_client_list(self):
        """Refresh available clients for log viewing"""
        try:
            # Get running client containers
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=client', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                clients = [c.strip() for c in result.stdout.strip().split('\n') if c.strip()]
                
                # Update client selector dropdown
                current_client = self.client_selector.currentData()
                self.client_selector.clear()
                
                if clients:
                    for client in clients:
                        # Extract client number from container name
                        client_label = client.replace('_', ' ').title()
                        self.client_selector.addItem(f"🖥️ {client_label}", client)
                    
                    self.output_text.append(f"✅ Detected {len(clients)} client(s) for log viewing\n")
                else:
                    self.client_selector.addItem("No clients detected", None)
                    self.output_text.append("⚠️ No client containers detected\n")
            
        except Exception as e:
            self.output_text.append(f"⚠️ Could not detect clients: {e}\n")
            self.client_selector.clear()
            self.client_selector.addItem("Error detecting clients", None)
    
    def switch_client_log(self):
        """Switch to viewing logs of a different client"""
        client_name = self.client_selector.currentData()
        
        if client_name is None:
            return
        
        # Stop current client log monitor if running
        if hasattr(self, 'client_log_monitor') and self.client_log_monitor.isRunning():
            self.client_log_monitor.stop()
            self.client_log_monitor.wait()
        
        # Clear current log display
        self.client_log_text.clear()
        self.client_log_text.append(f"=== Logs for {client_name} ===\n")
        
        # Start new log monitor for selected client
        self.client_log_monitor = LogMonitor(client_name, "client")
        self.client_log_monitor.log_update.connect(
            lambda log_type, text: self.client_log_text.append(text)
        )
        self.client_log_monitor.start()

    
    def get_selected_use_case(self):
        """Get selected use case"""
        for button in self.use_case_group.button_group.buttons():
            if button.isChecked():
                return button.property("value")
        return "emotion"
    
    def get_selected_protocols(self):
        """Get selected protocols"""
        protocols = []
        for checkbox in self.protocol_checkboxes:
            if checkbox.isChecked():
                protocols.append(checkbox.property("value"))
        return protocols
    
    def get_selected_scenarios(self):
        """Get selected scenarios"""
        scenarios = []
        for checkbox in self.scenario_checkboxes:
            if checkbox.isChecked():
                scenarios.append(checkbox.property("value"))
        return scenarios
    
    def build_command(self):
        """Build the experiment command"""
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        script = f"{base_dir}/Network_Simulation/run_network_experiments.py"
        
        # Basic parameters
        cmd_parts = ["python3", script]
        
        # Use case
        use_case = self.get_selected_use_case()
        cmd_parts.extend(["--use-case", use_case])
        
        # GPU (always enabled for baseline)
        if self.gpu_enabled.isChecked():
            cmd_parts.append("--enable-gpu")
        
        # Baseline mode flag
        if self.baseline_mode.isChecked():
            cmd_parts.append("--baseline")
        
        # Rounds
        cmd_parts.extend(["--rounds", str(self.rounds_spinbox.value())])
        
        # Protocols
        protocols = self.get_selected_protocols()
        if protocols:
            cmd_parts.extend(["--protocols"] + protocols)
        
        # For baseline mode, skip network scenarios (script handles it automatically)
        if not self.baseline_mode.isChecked():
            # Scenarios
            scenarios = self.get_selected_scenarios()
            if scenarios:
                cmd_parts.extend(["--scenarios"] + scenarios)
            
            # Network parameters (if custom values set)
            if self.latency_slider.value() > 0:
                cmd_parts.extend(["--latency", str(self.latency_slider.value())])
            if self.bandwidth_slider.value() != 100:
                cmd_parts.extend(["--bandwidth", str(self.bandwidth_slider.value())])
            if self.jitter_slider.value() > 0:
                cmd_parts.extend(["--jitter", str(self.jitter_slider.value())])
            if self.packet_loss_slider.value() > 0:
                cmd_parts.extend(["--packet-loss", str(self.packet_loss_slider.value())])
            
            # Congestion
            if self.enable_congestion.isChecked():
                cmd_parts.append("--enable-congestion")
                cmd_parts.extend(["--congestion-level", self.congestion_level.currentText().lower()])
        
        # Quantization
        if self.quantization_enabled.isChecked():
            cmd_parts.append("--use-quantization")
            cmd_parts.extend(["--quantization-bits", self.quant_bits.currentText()])
            cmd_parts.extend(["--quantization-strategy", self.quant_strategy.currentText()])
            if self.quant_symmetric.isChecked():
                cmd_parts.append("--quantization-symmetric")
        
        # Compression
        if self.compression_enabled.isChecked():
            cmd_parts.append("--enable-compression")
            cmd_parts.extend(["--compression-algorithm", self.compression_algo.currentText()])
            cmd_parts.extend(["--compression-level", str(self.compression_level.value())])
        
        # Q-learning convergence (only when rl_unified is selected)
        if self.use_ql_convergence.isChecked() and "rl_unified" in self.get_selected_protocols():
            cmd_parts.append("--use-ql-convergence")
        
        return " ".join(cmd_parts)
    
    def start_experiment(self):
        """Start the experiment"""
        # Validate baseline mode
        if self.baseline_mode.isChecked():
            if not self.gpu_enabled.isChecked():
                QMessageBox.critical(
                    self, 
                    "Baseline Mode Error", 
                    "GPU must be enabled for baseline mode!\n"
                    "Baseline models require GPU for proper training."
                )
                return
            
            # Confirm baseline mode
            reply = QMessageBox.question(
                self,
                "Confirm Baseline Mode",
                "🎯 BASELINE MODE ENABLED\n\n"
                "This will create reference models with:\n"
                "  • Excellent network conditions (no latency/packet loss)\n"
                "  • GPU acceleration (forced)\n"
                "  • Saved to experiment_results_baseline/\n\n"
                "These will be used for comparison in future experiments.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Validate selections
        if not self.get_selected_protocols():
            QMessageBox.warning(self, "Warning", "Please select at least one protocol!")
            return
        
        # Validate scenarios (only for non-baseline mode)
        if not self.baseline_mode.isChecked() and not self.get_selected_scenarios():
            QMessageBox.warning(self, "Warning", "Please select at least one network scenario!")
            return
        
        # Build command
        command = self.build_command()
        
        # Confirm
        mode_str = "BASELINE" if self.baseline_mode.isChecked() else "NETWORK EXPERIMENT"
        scenarios_str = "N/A (Excellent Network)" if self.baseline_mode.isChecked() else ', '.join(self.get_selected_scenarios())
        
        reply = QMessageBox.question(
            self,
            "Confirm Experiment",
            f"Ready to start {mode_str} with:\n\n"
            f"Mode: {mode_str}\n"
            f"Use Case: {self.get_selected_use_case()}\n"
            f"Protocols: {', '.join(self.get_selected_protocols())}\n"
            f"Scenarios: {', '.join(self.get_selected_scenarios())}\n"
            f"Rounds: {self.rounds_spinbox.value()}\n"
            f"GPU: {'Enabled' if self.gpu_enabled.isChecked() else 'Disabled'}\n\n"
            f"Command:\n{command}\n\n"
            f"This may take a long time. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        self._run_command_common(command)
    
    def start_diagnostic_pipeline(self):
        """Run diagnostic pipeline for the single selected protocol (separate from normal FL run)."""
        protocols = self.get_selected_protocols()
        pipeline_protocols = ["mqtt", "amqp", "grpc", "quic", "http3", "dds"]
        if len(protocols) != 1:
            QMessageBox.warning(
                self,
                "Diagnostic Pipeline",
                "Please select exactly ONE protocol for the diagnostic pipeline.\n\n"
                "Supported: MQTT, AMQP, gRPC, QUIC, HTTP/3, DDS."
            )
            return
        protocol = protocols[0].lower()
        if protocol not in pipeline_protocols:
            QMessageBox.warning(
                self,
                "Diagnostic Pipeline",
                f"Diagnostic pipeline supports only: MQTT, AMQP, gRPC, QUIC, HTTP/3, DDS.\n"
                f"Selected: {protocol}. RL-Unified is not supported."
            )
            return
        scenarios = self.get_selected_scenarios()
        if len(scenarios) != 1:
            QMessageBox.warning(
                self,
                "Diagnostic Pipeline",
                "Please select exactly ONE network scenario for the diagnostic pipeline.\n\n"
                "Phase 1 uses a clean channel (no losses). Phases 2–4 use the selected scenario."
            )
            return
        scenario = scenarios[0].lower()
        use_case = self.get_selected_use_case()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(base_dir, "Network_Simulation", "diagnostic_pipeline.py")
        command = f"python3 {script} --protocol {protocol} --use-case {use_case} --scenario {scenario}"
        if self.gpu_enabled.isChecked():
            command += " --enable-gpu"
        reply = QMessageBox.question(
            self,
            "Run Diagnostic Pipeline",
            f"Run diagnostic pipeline for:\n\n"
            f"  Protocol: {protocol.upper()}\n"
            f"  Use Case: {use_case}\n"
            f"  Network scenario (Phases 2–4): {scenario}\n"
            f"  GPU: {'Enabled (fallback to CPU if unavailable)' if self.gpu_enabled.isChecked() else 'Disabled'}\n\n"
            f"  Phase 1: Calibration with NO channel losses (protocol & broker overhead only).\n"
            f"  Phases 2–4: Apply scenario '{scenario}' → extract tc → lossy round → table.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        self._run_command_common(command)
    
    def _run_command_common(self, command):
        """Shared logic to run a command in the experiment thread."""
        # Clear all output
        self.clear_all_output()
        self.experiment_started_at = datetime.now()
        self.current_results_dir = None
        
        # Update UI
        self.start_button.setEnabled(False)
        if hasattr(self, 'diagnostic_pipeline_button'):
            self.diagnostic_pipeline_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.show()
        self.statusBar().showMessage("🚀 Experiment running...")
        
        # Start experiment thread
        self.experiment_thread = ExperimentRunner(command)
        self.experiment_thread.progress_update.connect(self.update_output)
        self.experiment_thread.experiment_finished.connect(self.experiment_completed)
        self.experiment_thread.start()
        
        # Start FL training dashboard monitoring
        self.start_dashboard_monitor()
        
        # Diagnostic pipeline may do GPU fail + CPU compose + 15s sleep; delay first container detection so containers are up
        is_diagnostic = "diagnostic_pipeline" in (command or "")
        first_delay_ms = 22000 if is_diagnostic else 5000
        first_refresh_ms = first_delay_ms + 2000
        
        # Start container log monitoring (after delay so containers are running)
        QTimer.singleShot(first_delay_ms, self.start_log_monitors)
        
        # Add periodic retry for log monitors in case containers start late
        QTimer.singleShot(15000, self.retry_log_monitors_if_needed)
        QTimer.singleShot(30000, self.retry_log_monitors_if_needed)
        if is_diagnostic:
            QTimer.singleShot(45000, self.retry_log_monitors_if_needed)
        
        # Refresh server and client lists for both network targeting and log viewing
        QTimer.singleShot(first_refresh_ms, self.refresh_server_list)
        QTimer.singleShot(first_refresh_ms, self.refresh_client_targets)
        QTimer.singleShot(first_refresh_ms, self.refresh_client_list)
    
    def stop_experiment(self):
        """Stop the running experiment"""
        if self.experiment_thread:
            reply = QMessageBox.question(
                self,
                "Stop Experiment",
                "Are you sure you want to stop the running experiment?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.output_text.append("\n🛑 Stopping experiment...\n")
                self.statusBar().showMessage("Stopping experiment...")
                
                # Stop the experiment thread
                self.experiment_thread.stop()
                self.experiment_thread.wait(5000)  # Wait up to 5 seconds

                # Save final container logs before monitors/containers are stopped
                self.save_container_logs_snapshot()
                
                # Stop monitors
                self.stop_all_monitors()
                
                # Stop and remove all FL containers
                self.output_text.append("🗑️ Cleaning up containers...\n")
                try:
                    subprocess.run(
                        ["docker", "ps", "-a", "--filter", "name=fl-", "--format", "{{.Names}}"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    containers = [c.strip() for c in subprocess.run(
                        ["docker", "ps", "-a", "--filter", "name=fl-", "--format", "{{.Names}}"],
                        capture_output=True, text=True, timeout=10
                    ).stdout.split('\n') if c.strip()]
                    
                    if containers:
                        self.output_text.append(f"  Stopping {len(containers)} containers...\n")
                        subprocess.run(["docker", "stop"] + containers, timeout=30)
                        self.output_text.append(f"  Removing {len(containers)} containers...\n")
                        subprocess.run(["docker", "rm"] + containers, timeout=30)
                        self.output_text.append(f"✅ Cleaned up {len(containers)} containers\n")
                    
                except Exception as e:
                    self.output_text.append(f"⚠️ Cleanup error: {e}\n")
                
                # Reset UI
                self.reset_ui()
                self.output_text.append("\n✅ Experiment stopped successfully\n")
                self.statusBar().showMessage("Experiment stopped")
        else:
            QMessageBox.information(self, "No Experiment", "No experiment is currently running.")
    
    def start_dashboard_monitor(self):
        """Start FL training dashboard monitoring"""
        use_case = self.get_selected_use_case()
        self.dashboard_text.append(f"🚀 Starting FL Training Dashboard for {use_case}...\n")
        
        self.dashboard_thread = DashboardMonitor(use_case)
        self.dashboard_thread.dashboard_update.connect(self.update_dashboard)
        self.dashboard_thread.start()
    
    def start_log_monitors(self):
        """Start monitoring server and client container logs"""
        # IMPORTANT: Stop any existing monitors first to avoid duplicates
        # This ensures clean restart when running multiple experiments
        for monitor in self.log_monitors:
            try:
                monitor.stop()
                monitor.wait(1000)  # Wait up to 1 second
            except:
                pass
        self.log_monitors.clear()
        
        # Refresh server and client lists first to populate dropdowns
        if hasattr(self, 'refresh_server_list'):
            self.refresh_server_list()
        if hasattr(self, 'refresh_client_list'):
            self.refresh_client_list()
        
        # Get running containers
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
                
                # Find server and client containers
                server_containers = [c for c in containers if 'server' in c.lower() and 'fl-' in c]
                client_containers = [c for c in containers if 'client' in c.lower() and 'fl-' in c]
                
                # Monitor first server container (or use selected server if available)
                if server_containers:
                    # Use selected server from dropdown if available, otherwise use first
                    selected_server = self.server_selector.currentData() if hasattr(self, 'server_selector') else None
                    server_to_monitor = selected_server if selected_server and selected_server in server_containers else server_containers[0]
                    
                    server_monitor = LogMonitor(server_to_monitor, "server")
                    server_monitor.log_update.connect(self.update_logs)
                    server_monitor.start()
                    self.log_monitors.append(server_monitor)
                    self.server_log_text.append(f"📡 Monitoring: {server_to_monitor}\n")
                else:
                    self.server_log_text.append(f"⚠️ No server containers found. Waiting...\n")
                
                # Monitor first client container
                if client_containers:
                    client_monitor = LogMonitor(client_containers[0], "client")
                    client_monitor.log_update.connect(self.update_logs)
                    client_monitor.start()
                    self.log_monitors.append(client_monitor)
                    self.client_log_text.append(f"📡 Monitoring: {client_containers[0]}\n")
                else:
                    self.client_log_text.append(f"⚠️ No client containers found. Waiting...\n")
                    
        except Exception as e:
            self.server_log_text.append(f"⚠️ Error starting log monitors: {str(e)}\n")
    
    def retry_log_monitors_if_needed(self):
        """Retry starting log monitors if no active monitors are running"""
        # Check if any log monitors are actually receiving data
        active_monitors = [m for m in self.log_monitors if m.isRunning()]
        
        if len(active_monitors) < 2:  # We expect server + client monitors
            self.server_log_text.append(f"\n🔄 Retrying log monitor setup (found {len(active_monitors)} active monitors)...\n")
            self.start_log_monitors()
    
    def stop_all_monitors(self):
        """Stop all monitoring threads"""
        if self.dashboard_thread:
            self.dashboard_thread.stop()
            self.dashboard_thread.wait()
            self.dashboard_thread = None
        
        for monitor in self.log_monitors:
            monitor.stop()
            monitor.wait()
        self.log_monitors.clear()
    
    def apply_network_conditions(self):
        """Apply network conditions using fl_network_monitor.py"""
        # Check if baseline mode is active
        if self.baseline_mode.isChecked():
            QMessageBox.warning(
                self,
                "Baseline Mode Active",
                "Network conditions cannot be changed in baseline mode.\n"
                "Baseline requires excellent network conditions for accurate reference models."
            )
            return
        
        latency = self.latency_slider.value()
        bandwidth = self.bandwidth_slider.value()
        jitter = self.jitter_slider.value()
        packet_loss = self.packet_loss_slider.value()
        
        if latency == 0 and bandwidth == 100 and jitter == 0 and packet_loss == 0:
            QMessageBox.information(
                self,
                "No Changes",
                "No network conditions to apply. Adjust sliders first."
            )
            return
        
        # Get selected target
        target = self.network_target.currentData()
        target_name = self.network_target.currentText()
        
        self.output_text.append(
            f"\n🌐 Applying network conditions to {target_name}:\n"
            f"  Latency: {latency}ms\n"
            f"  Bandwidth: {bandwidth}Mbps\n"
            f"  Jitter: {jitter}ms\n"
            f"  Packet Loss: {packet_loss}%\n"
        )
        
        # Pass target to network controller
        self.network_controller = NetworkController(
            latency, bandwidth, jitter, packet_loss, target
        )
        self.network_controller.control_update.connect(self.update_output)
        self.network_controller.start()
    
    def update_dashboard(self, text):
        """Update dashboard monitor output"""
        self.dashboard_text.insertPlainText(text)
        self.dashboard_text.verticalScrollBar().setValue(
            self.dashboard_text.verticalScrollBar().maximum()
        )
    
    def update_logs(self, log_type, text):
        """Update container logs"""
        if log_type == "server":
            self.server_log_text.insertPlainText(text)
            self.server_log_text.verticalScrollBar().setValue(
                self.server_log_text.verticalScrollBar().maximum()
            )
        elif log_type == "client":
            self.client_log_text.insertPlainText(text)
            self.client_log_text.verticalScrollBar().setValue(
                self.client_log_text.verticalScrollBar().maximum()
            )
    
    def clear_all_output(self):
        """Clear all output consoles and prepare for fresh experiment"""
        self.output_text.clear()
        self.dashboard_text.clear()
        self.server_log_text.clear()
        self.client_log_text.clear()
        if hasattr(self, "_diag_json_buffer"):
            self._diag_json_buffer = ""
        # Add informative message
        self.server_log_text.append("📋 Server logs cleared. Waiting for new experiment to start...\n")
        self.client_log_text.append("📋 Client logs cleared. Waiting for new experiment to start...\n")
    
    def update_output(self, text):
        """Update output console"""
        self._extract_results_dir_from_output(text)
        # Parse diagnostic pipeline JSON (may arrive in one chunk or split across chunks)
        if "FL_DIAG_TABLE_JSON|" in text:
            json_str = text.split("FL_DIAG_TABLE_JSON|", 1)[1]
            if not hasattr(self, "_diag_json_buffer"):
                self._diag_json_buffer = ""
            self._diag_json_buffer = json_str
        elif getattr(self, "_diag_json_buffer", None):
            self._diag_json_buffer += text
        else:
            self._diag_json_buffer = ""
        if getattr(self, "_diag_json_buffer", None):
            buf = self._diag_json_buffer.strip()
            if buf:
                summary = None
                try:
                    summary = json.loads(buf)
                except json.JSONDecodeError:
                    # Extract JSON array by matching brackets (handles trailing text or split chunks)
                    start = buf.find("[")
                    if start >= 0:
                        depth = 0
                        for i, c in enumerate(buf[start:], start=start):
                            if c == "[":
                                depth += 1
                            elif c == "]":
                                depth -= 1
                                if depth == 0:
                                    try:
                                        summary = json.loads(buf[start : i + 1])
                                    except json.JSONDecodeError:
                                        pass
                                    break
                if summary is not None:
                    self.update_diagnostic_results_table(summary)
                    self._diag_json_buffer = ""
        self.output_text.insertPlainText(text)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def update_diagnostic_results_table(self, summary):
        """Update Diagnostic Results tab: merge by (protocol, scenario).
        - Same protocol + same scenario: replace those rows with the new run.
        - Same protocol + different scenario, or different protocol: append new rows."""
        if not hasattr(self, "diagnostic_results_table"):
            return
        from PyQt5.QtWidgets import QTableWidgetItem
        if not hasattr(self, "_diagnostic_results_cache"):
            self._diagnostic_results_cache = []
        new_rows = summary if isinstance(summary, list) else [summary]
        # Keys (protocol, scenario) that this run is updating
        updated_keys = set()
        for r in new_rows:
            updated_keys.add((str(r.get("protocol", "")).strip(), str(r.get("scenario", "")).strip()))
        # Remove existing rows that match any (protocol, scenario) we are updating
        self._diagnostic_results_cache = [
            row for row in self._diagnostic_results_cache
            if (str(row.get("protocol", "")).strip(), str(row.get("scenario", "")).strip()) not in updated_keys
        ]
        # Append new rows
        self._diagnostic_results_cache.extend(new_rows)
        # Render full table
        rows = self._diagnostic_results_cache
        self.diagnostic_results_table.setRowCount(len(rows))
        for row_idx, r in enumerate(rows):
            self.diagnostic_results_table.setItem(row_idx, 0, QTableWidgetItem(str(r.get("client_id", row_idx + 1))))
            self.diagnostic_results_table.setItem(row_idx, 1, QTableWidgetItem(str(r.get("protocol", ""))))
            self.diagnostic_results_table.setItem(row_idx, 2, QTableWidgetItem(str(r.get("scenario", ""))))
            self.diagnostic_results_table.setItem(row_idx, 3, QTableWidgetItem(f"{r.get('O_app', 0):.6f}"))
            self.diagnostic_results_table.setItem(row_idx, 4, QTableWidgetItem(f"{r.get('O_broker', 0):.6f}"))
            self.diagnostic_results_table.setItem(row_idx, 5, QTableWidgetItem(f"{r.get('p', 0):.4f}"))
            self.diagnostic_results_table.setItem(row_idx, 6, QTableWidgetItem(f"{r.get('T_actual', 0):.4f}"))
            self.diagnostic_results_table.setItem(row_idx, 7, QTableWidgetItem(f"{r.get('T_calc', 0):.4f}"))
            self.diagnostic_results_table.setItem(row_idx, 8, QTableWidgetItem(f"{r.get('error_pct', 0):.2f}%"))
        for i in range(self.output_tabs.count()):
            if self.output_tabs.tabText(i) == "📊 Diagnostic Results":
                self.output_tabs.setCurrentIndex(i)
                break

    def _extract_results_dir_from_output(self, text):
        """Track results directory from runner stdout."""
        markers = ("Results Directory:", "Results saved in:")
        for marker in markers:
            if marker in text:
                path_part = text.split(marker, 1)[1].strip()
                if path_part:
                    self.current_results_dir = path_part
                return

    def _get_fallback_results_dir(self):
        """Best-effort results directory when parser didn't capture it yet."""
        base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
        use_case = self.get_selected_use_case()

        if self.baseline_mode.isChecked():
            candidate = os.path.join(base_dir, "experiment_results_baseline", use_case)
            return candidate if os.path.isdir(candidate) else None

        candidate_root = os.path.join(base_dir, "experiment_results")
        if not os.path.isdir(candidate_root):
            return None

        use_case_prefix = f"{use_case}_"
        candidates = [
            os.path.join(candidate_root, d)
            for d in os.listdir(candidate_root)
            if d.startswith(use_case_prefix) and os.path.isdir(os.path.join(candidate_root, d))
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _get_log_capture_dir(self):
        """Resolve folder where stop-time container logs should be written."""
        base_results = self.current_results_dir
        if not base_results or not os.path.isdir(base_results):
            base_results = self._get_fallback_results_dir()

        if not base_results:
            base_dir = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
            base_results = os.path.join(base_dir, "experiment_results", "manual_stop_logs")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = os.path.join(base_results, "container_logs", f"stopped_{stamp}")
        os.makedirs(target, exist_ok=True)
        return target

    def save_container_logs_snapshot(self):
        """Save server/client/broker logs into experiment results on Stop."""
        try:
            log_dir = self._get_log_capture_dir()
            self.output_text.append(f"💾 Saving container logs to: {log_dir}\n")

            ps_cmd = ["docker", "ps", "-a", "--format", "{{.Names}}"]
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.output_text.append("⚠️ Could not list containers for log export\n")
                return

            all_names = [c.strip() for c in result.stdout.split('\n') if c.strip()]
            keywords = ("server", "client", "broker", "rabbitmq", "mosquitto")
            log_targets = [
                name for name in all_names
                if name.startswith("fl-") or any(k in name.lower() for k in keywords)
            ]

            if not log_targets:
                self.output_text.append("⚠️ No server/client/broker containers found for log export\n")
                return

            saved = 0
            for container in sorted(set(log_targets)):
                safe_name = container.replace("/", "_")
                out_file = os.path.join(log_dir, f"{safe_name}.log")
                log_cmd = ["docker", "logs", "--timestamps", container]
                log_res = subprocess.run(log_cmd, capture_output=True, text=True, timeout=20)

                with open(out_file, "w", encoding="utf-8") as f:
                    if log_res.stdout:
                        f.write(log_res.stdout)
                    if log_res.stderr:
                        f.write("\n[stderr]\n")
                        f.write(log_res.stderr)
                saved += 1

            meta_file = os.path.join(log_dir, "log_capture_metadata.txt")
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(f"captured_at={datetime.now().isoformat()}\n")
                f.write(f"use_case={self.get_selected_use_case()}\n")
                f.write(f"containers_saved={saved}\n")
                f.write("targets=\n")
                for c in sorted(set(log_targets)):
                    f.write(f"- {c}\n")

            self.output_text.append(f"✅ Saved logs for {saved} container(s)\n")
        except Exception as e:
            self.output_text.append(f"⚠️ Failed to save stop-time logs: {e}\n")
    
    def experiment_completed(self, success, message):
        """Handle experiment completion"""
        self.update_output(f"\n\n{message}\n")
        # Ensure Diagnostic Results tab is populated when diagnostic pipeline ran (stdout or file fallback)
        if hasattr(self, "diagnostic_results_table"):
            summary = None
            full = self.output_text.toPlainText()
            if "FL_DIAG_TABLE_JSON|" in full:
                try:
                    after = full.split("FL_DIAG_TABLE_JSON|", 1)[1]
                    # Extract JSON array: find first '[' and matching ']' so chunking/trailing text doesn't break parse
                    start = after.find("[")
                    if start >= 0:
                        depth = 0
                        for i, c in enumerate(after[start:], start=start):
                            if c == "[":
                                depth += 1
                            elif c == "]":
                                depth -= 1
                                if depth == 0:
                                    summary = json.loads(after[start : i + 1])
                                    break
                except (json.JSONDecodeError, IndexError, ValueError):
                    pass
            if summary is None:
                # File fallback: pipeline writes to shared_data/diagnostic_results_latest.json
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                diag_file = os.path.join(base_dir, "shared_data", "diagnostic_results_latest.json")
                try:
                    if os.path.isfile(diag_file):
                        with open(diag_file, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
            if summary is not None:
                self.update_diagnostic_results_table(summary)
        self.stop_all_monitors()
        self.reset_ui()
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.statusBar().showMessage("✅ Experiment completed successfully")
        else:
            QMessageBox.warning(self, "Error", message)
            self.statusBar().showMessage("❌ Experiment failed")
    
    def reset_ui(self):
        """Reset UI after experiment"""
        self.start_button.setEnabled(True)
        if hasattr(self, 'diagnostic_pipeline_button'):
            self.diagnostic_pipeline_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.hide()
        if self.statusBar().currentMessage().startswith("🚀"):
            self.statusBar().showMessage("Ready to run experiments")
    
    def clear_output(self):
        """Clear output console (legacy - use clear_all_output)"""
        self.clear_all_output()
    
    def closeEvent(self, event):
        """Handle window close - stop all threads"""
        if self.experiment_thread and self.experiment_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Experiment Running",
                "An experiment is still running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.experiment_thread:
                    self.experiment_thread.stop()
                    self.experiment_thread.wait()
                self.stop_all_monitors()
                event.accept()
            else:
                event.ignore()
        else:
            self.stop_all_monitors()
            event.accept()
    
    def get_stylesheet(self):
        """Get application stylesheet"""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #333;
            }
            QLabel {
                color: #333;
                font-size: 12px;
            }
            QComboBox, QSpinBox, QLineEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                min-width: 100px;
            }
            QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
                border-color: #667eea;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                border: 1px solid #5568d3;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5568d3;
            }
            QCheckBox, QRadioButton {
                spacing: 8px;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #667eea;
                border: 2px solid #5568d3;
                border-radius: 3px;
            }
            QRadioButton::indicator:checked {
                background-color: #667eea;
                border: 3px solid white;
                border-radius: 9px;
            }
            QTabWidget::pane {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #667eea;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 6px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #667eea;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #999;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
        """
    
    def get_group_style(self):
        """Get group box style"""
        return """
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #333;
            }
        """


def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show GUI
    gui = FLExperimentGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
