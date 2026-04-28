#!/usr/bin/env python3
"""
Plot cumulative FL wall time, accuracy, and loss from client metrics JSONL file for all protocols.
"""

import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def plot_metric_all_protocols(jsonl_path: str, metric_name: str, output_path: str = None, 
                               ylabel: str = None, title: str = None):
    """
    Read JSONL file and plot a metric for all protocols.
    
    Args:
        jsonl_path: Path to the client metrics JSONL file
        metric_name: Name of the metric field to plot
        output_path: Optional path to save the plot (if None, displays interactively)
        ylabel: Y-axis label
        title: Plot title
    """
    # Read JSONL file and group by protocol
    protocol_data = defaultdict(lambda: {'rounds': [], 'values': []})
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                protocol = data.get('protocol', 'unknown')
                round_num = data['round']
                
                # Get the metric value
                if metric_name in data:
                    metric_value = data[metric_name]
                    protocol_data[protocol]['rounds'].append(round_num)
                    protocol_data[protocol]['values'].append(metric_value)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each protocol (all using same small circle marker)
    protocol_styles = {
        'mqtt': {'color': '#1f77b4', 'marker': 'o', 'label': 'MQTT'},
        'amqp': {'color': '#ff7f0e', 'marker': 'o', 'label': 'AMQP'},
        'grpc': {'color': '#2ca02c', 'marker': 'o', 'label': 'gRPC'},
        'http3': {'color': '#d62728', 'marker': 'o', 'label': 'HTTP/3'},
        'unified': {'color': '#9467bd', 'marker': 'o', 'label': 'RL Unified'},
    }
    
    # Plot data for each protocol
    for protocol in ['mqtt', 'amqp', 'grpc', 'http3', 'unified']:
        if protocol in protocol_data:
            data = protocol_data[protocol]
            style = protocol_styles.get(protocol, {'color': 'gray', 'marker': 'o', 'label': protocol})
            
            ax.plot(data['rounds'], data['values'], 
                   marker=style['marker'], markersize=4, 
                   linewidth=2, color=style['color'], 
                   label=style['label'])
    
    # Set labels and title
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel or metric_name, fontsize=12, fontweight='bold')
    ax.set_title(title or 'Federated learning - training results comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {metric_name}:")
    for protocol in ['mqtt', 'amqp', 'grpc', 'http3', 'unified']:
        if protocol in protocol_data:
            data = protocol_data[protocol]
            if data['values']:
                final_value = data['values'][-1]
                avg_value = sum(data['values']) / len(data['values'])
                print(f"\n{protocol.upper()}:")
                print(f"  Total rounds: {len(data['rounds'])}")
                print(f"  Final {metric_name}: {final_value:.4f}")
                print(f"  Average {metric_name}: {avg_value:.4f}")


def plot_cumulative_fl_wall_time_all_protocols(jsonl_path: str, output_path: str = None):
    """Plot cumulative FL wall time for all protocols."""
    plot_metric_all_protocols(
        jsonl_path=jsonl_path,
        metric_name='total_fl_cumulative_wall_time_sec',
        output_path=output_path,
        ylabel='Cumulative FL Wall Time (seconds)',
        title='Federated learning - training results comparison'
    )


def plot_accuracy_all_protocols(jsonl_path: str, output_path: str = None):
    """Plot accuracy for all protocols."""
    plot_metric_all_protocols(
        jsonl_path=jsonl_path,
        metric_name='accuracy',
        output_path=output_path,
        ylabel='Accuracy',
        title='Federated learning - training results comparison'
    )


def plot_loss_all_protocols(jsonl_path: str, output_path: str = None):
    """Plot loss for all protocols."""
    plot_metric_all_protocols(
        jsonl_path=jsonl_path,
        metric_name='loss',
        output_path=output_path,
        ylabel='Loss',
        title='Federated learning - training results comparison'
    )


def plot_battery_model_consumption_from_json(json_files: dict, output_path: str = None):
    """
    Plot battery model consumption from training results JSON files.
    
    Args:
        json_files: Dictionary mapping protocol names to JSON file paths
        output_path: Optional path to save the plot
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each protocol (all using same small circle marker)
    protocol_styles = {
        'mqtt': {'color': '#1f77b4', 'marker': 'o', 'label': 'MQTT'},
        'amqp': {'color': '#ff7f0e', 'marker': 'o', 'label': 'AMQP'},
        'grpc': {'color': '#2ca02c', 'marker': 'o', 'label': 'gRPC'},
        'http3': {'color': '#d62728', 'marker': 'o', 'label': 'HTTP/3'},
        'unified': {'color': '#9467bd', 'marker': 'o', 'label': 'RL Unified'},
    }
    
    # Plot data for each protocol
    for protocol in ['mqtt', 'amqp', 'grpc', 'http3', 'unified']:
        if protocol in json_files:
            # Read JSON file
            with open(json_files[protocol], 'r') as f:
                data = json.load(f)
            
            rounds = data.get('rounds', [])
            battery_consumption = data.get('battery_model_consumption', [])
            
            if rounds and battery_consumption:
                style = protocol_styles.get(protocol, {'color': 'gray', 'marker': 'o', 'label': protocol})
                
                ax.plot(rounds, battery_consumption, 
                       marker=style['marker'], markersize=4, 
                       linewidth=2, color=style['color'], 
                       label=style['label'])
    
    # Set labels and title
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Battery model consumption', fontsize=12, fontweight='bold')
    ax.set_title('Federated learning - training results comparison', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics for Battery Model Consumption:")
    for protocol in ['mqtt', 'amqp', 'grpc', 'http3', 'unified']:
        if protocol in json_files:
            with open(json_files[protocol], 'r') as f:
                data = json.load(f)
            battery_consumption = data.get('battery_model_consumption', [])
            if battery_consumption:
                final_value = battery_consumption[-1]
                avg_value = sum(battery_consumption) / len(battery_consumption)
                print(f"\n{protocol.upper()}:")
                print(f"  Total rounds: {len(battery_consumption)}")
                print(f"  Final battery consumption: {final_value:.6e}")
                print(f"  Average battery consumption: {avg_value:.6e}")


if __name__ == "__main__":
    # Default path - using the temperature client2 data from shared_data
    jsonl_file = "shared_data/client_fl_metrics_temperature_client2.jsonl"
    
    # Create all three plots from JSONL
    print("=" * 60)
    print("Generating Cumulative FL Wall Time plot...")
    print("=" * 60)
    plot_cumulative_fl_wall_time_all_protocols(jsonl_file, "cumulative_fl_wall_time_all_protocols.png")
    
    print("\n" + "=" * 60)
    print("Generating Accuracy plot...")
    print("=" * 60)
    plot_accuracy_all_protocols(jsonl_file, "accuracy_all_protocols.png")
    
    print("\n" + "=" * 60)
    print("Generating Loss plot...")
    print("=" * 60)
    plot_loss_all_protocols(jsonl_file, "loss_all_protocols.png")
    
    # Battery model consumption from training results JSON files
    print("\n" + "=" * 60)
    print("Generating Battery Model Consumption plot...")
    print("=" * 60)
    
    battery_json_files = {
        'mqtt': 'experiment_results/temperature_20260420_105727/mqtt_dynamic/mqtt_training_results.json',
        'amqp': 'experiment_results/temperature_20260420_110115/amqp_dynamic/amqp_training_results.json',
        'grpc': 'experiment_results/temperature_20260420_110627/grpc_dynamic/grpc_training_results.json',
        'http3': 'experiment_results/temperature_20260420_111118/http3_dynamic/http3_training_results.json',
        'unified': 'experiment_results/temperature_20260421_115332/rl_unified_dynamic/rl_unified_training_results.json',
    }
    
    plot_battery_model_consumption_from_json(battery_json_files, "battery_model_consumption_all_protocols.png")
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)

