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
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🚀 Federated Learning Network Experiment Dashboard")
        self.setGeometry(100, 100, 1400, 900)
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
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)
        
        # Tab widget for different configuration sections
        self.config_tabs = QTabWidget()
        self.config_tabs.addTab(self.create_basic_config_tab(), "⚙️ Basic Configuration")
        self.config_tabs.addTab(self.create_network_config_tab(), "🌐 Network Control")
        self.config_tabs.addTab(self.create_advanced_config_tab(), "🔧 Advanced Options")
        self.config_tabs.addTab(self.create_docker_build_tab(), "🐳 Docker Build")
        config_layout.addWidget(self.config_tabs)
        splitter.addWidget(config_widget)
        
        # Bottom section: Monitoring and Output (NEW!)
        monitor_output_widget = self.create_monitoring_output_section()
        splitter.addWidget(monitor_output_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready to run experiments")
        
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
    
    def create_basic_config_tab(self):
        """Create basic configuration tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(15)
        
        row = 0
        
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
        layout.addWidget(baseline_group, row, 0, 1, 2)
        row += 1
        
        # Use Case
        use_case_group = self.create_group("📊 Use Case", [
            ("Mental State Recognition", "mentalstate"),
            ("Emotion Recognition", "emotion"),
            ("Temperature Regulation", "temperature")
        ])
        layout.addWidget(use_case_group, row, 0, 1, 2)
        self.use_case_group = use_case_group
        row += 1
        
        # Protocol Selection
        protocol_group = self.create_checkbox_group("📡 Communication Protocols", [
            ("MQTT", "mqtt", True),
            ("AMQP", "amqp", False),
            ("gRPC", "grpc", False),
            ("QUIC", "quic", False),
            ("DDS", "dds", False),
            ("🤖 RL-Unified (Dynamic Selection)", "rl_unified", False)
        ])
        layout.addWidget(protocol_group, row, 0, 1, 2)
        self.protocol_checkboxes = protocol_group.findChildren(QCheckBox)
        row += 1
        
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
        layout.addWidget(scenario_group, row, 0, 1, 2)
        self.scenario_checkboxes = scenario_group.findChildren(QCheckBox)
        row += 1
        
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
        layout.addWidget(gpu_group, row, 0, 1, 2)
        row += 1
        
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
        layout.addWidget(training_group, row, 0, 1, 2)
        row += 1
        
        layout.setRowStretch(row, 1)
        return widget
    
    def create_network_config_tab(self):
        """Create network configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
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
        return widget
    
    def create_advanced_config_tab(self):
        """Create advanced configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
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
        return widget
    
    def create_docker_build_tab(self):
        """Create Docker build tab with build button"""
        import subprocess
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

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

        docker_group.setLayout(docker_layout)
        layout.addWidget(docker_group)

        layout.addStretch()
        return widget

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
        # Disable the button directly
        if hasattr(self, 'build_btn_emotion'):
            self.build_btn_emotion.setEnabled(False)
        self.docker_build_log_text.clear()
        self.output_tabs.setCurrentWidget(self.docker_build_log_text)
        # Start Docker build in a thread
        self.docker_build_thread = DockerBuildThread(cmd, base_dir)
        self.docker_build_thread.log_update.connect(self.docker_build_log_text.append)
        def on_build_finished(success, returncode):
            if hasattr(self, 'build_btn_emotion'):
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
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=1800)
            output = proc.stdout + "\n" + proc.stderr
            if proc.returncode == 0:
                self.statusBar().showMessage("✅ Docker images built successfully (Mental State)")
                QMessageBox.information(self, "Docker Build Complete", "Docker images for Mental State use case built successfully!\n\nOutput:\n" + output[-2000:])
            else:
                self.statusBar().showMessage("❌ Docker build failed (Mental State)")
                QMessageBox.critical(self, "Docker Build Failed", "Docker build failed!\n\nOutput:\n" + output[-2000:])
        except Exception as e:
            self.statusBar().showMessage("❌ Docker build error (Mental State)")
            QMessageBox.critical(self, "Docker Build Error", f"Error running Docker build:\n{e}")

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
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=1800)
            output = proc.stdout + "\n" + proc.stderr
            if proc.returncode == 0:
                self.statusBar().showMessage("✅ Docker images built successfully (Temperature Regulation)")
                QMessageBox.information(self, "Docker Build Complete", "Docker images for Temperature Regulation use case built successfully!\n\nOutput:\n" + output[-2000:])
            else:
                self.statusBar().showMessage("❌ Docker build failed (Temperature Regulation)")
                QMessageBox.critical(self, "Docker Build Failed", "Docker build failed!\n\nOutput:\n" + output[-2000:])
        except Exception as e:
            self.statusBar().showMessage("❌ Docker build error (Temperature Regulation)")
            QMessageBox.critical(self, "Docker Build Error", f"Error running Docker build:\n{e}")

    def create_monitoring_output_section(self):
        """Create comprehensive monitoring and output section, with Packet Logs tab"""
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Control buttons (unchanged)
        control_layout = QHBoxLayout()
        
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
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.apply_network_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)

        # Progress bar (unchanged)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Tabbed output section
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
        self.output_tabs.addTab(self.server_log_text, "🖥️ Server Logs")

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

        # Tab 5: Packet Logs
        packet_tab_widget = QWidget()
        packet_tab_layout = QVBoxLayout(packet_tab_widget)
        packet_tab_layout.setSpacing(5)
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
        self.output_tabs.addTab(packet_tab_widget, "📦 Packet Logs")
        QTimer.singleShot(100, self.refresh_packet_log_table)

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
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../packet_logs.db')
        db_path = os.path.abspath(db_path)
        self.packet_log_table.setRowCount(0)
        try:
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
        """Create a checkbox group"""
        group = QGroupBox(title)
        group.setStyleSheet(self.get_group_style())
        layout = QGridLayout()
        
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
                
                # Update network target dropdown
                current_value = self.network_target.currentData()
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
                
                self.output_text.append(f"✅ Detected {len(clients)} client container(s)\n")
            
        except Exception as e:
            self.output_text.append(f"⚠️ Could not detect client targets: {e}\n")
    
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
        
        # Clear all output
        self.clear_all_output()
        
        # Update UI
        self.start_button.setEnabled(False)
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
        
        # Start container log monitoring (after brief delay to let containers start)
        QTimer.singleShot(5000, self.start_log_monitors)
        
        # Refresh client lists for both network targeting and log viewing
        QTimer.singleShot(7000, self.refresh_client_targets)
        QTimer.singleShot(7000, self.refresh_client_list)
    
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
                self.experiment_thread.stop()
                self.experiment_thread.wait()
                self.stop_all_monitors()
                self.reset_ui()
    
    def start_dashboard_monitor(self):
        """Start FL training dashboard monitoring"""
        use_case = self.get_selected_use_case()
        self.dashboard_text.append(f"🚀 Starting FL Training Dashboard for {use_case}...\n")
        
        self.dashboard_thread = DashboardMonitor(use_case)
        self.dashboard_thread.dashboard_update.connect(self.update_dashboard)
        self.dashboard_thread.start()
    
    def start_log_monitors(self):
        """Start monitoring server and client container logs"""
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
                
                # Monitor first server container
                if server_containers:
                    server_monitor = LogMonitor(server_containers[0], "server")
                    server_monitor.log_update.connect(self.update_logs)
                    server_monitor.start()
                    self.log_monitors.append(server_monitor)
                    self.server_log_text.append(f"📡 Monitoring: {server_containers[0]}\n")
                
                # Monitor first client container
                if client_containers:
                    client_monitor = LogMonitor(client_containers[0], "client")
                    client_monitor.log_update.connect(self.update_logs)
                    client_monitor.start()
                    self.log_monitors.append(client_monitor)
                    self.client_log_text.append(f"📡 Monitoring: {client_containers[0]}\n")
                    
        except Exception as e:
            self.server_log_text.append(f"⚠️ Error starting log monitors: {str(e)}\n")
    
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
        """Clear all output consoles"""
        self.output_text.clear()
        self.dashboard_text.clear()
        self.server_log_text.clear()
        self.client_log_text.clear()
    
    def update_output(self, text):
        """Update output console"""
        self.output_text.insertPlainText(text)
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
    
    def experiment_completed(self, success, message):
        """Handle experiment completion"""
        self.update_output(f"\n\n{message}\n")
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
