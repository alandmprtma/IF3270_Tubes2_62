import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime
from train import train_and_evaluate_model
import config

# Create experiment results directory
RESULTS_DIR = "experiment_results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Get timestamp for unique experiment ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def run_layer_count_experiment():
    """Run experiment to test different numbers of RNN layers"""
    print("\n" + "="*50)
    print("EXPERIMENT: RNN LAYER COUNT VARIATION")
    print("="*50)
    
    results = []
    
    # For each layer count in our variations
    for num_layers in config.RNN_LAYERS_VARIATIONS:
        model_name = f"rnn_layers_{num_layers}"
        print(f"\nTesting model with {num_layers} RNN layer(s)")
        
        # Train model with this variation
        start_time = time.time()
        _, _, _, _, _, metrics = train_and_evaluate_model(
            num_rnn_layers=num_layers,
            model_name=model_name
        )
        train_time = time.time() - start_time
        
        # Store results
        results.append({
            'num_layers': num_layers,
            'accuracy': metrics['test_accuracy'],
            'f1_score': metrics['test_f1'],
            'loss': metrics['test_loss'],
            'train_time': train_time
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = f"{RESULTS_DIR}/layer_count_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Layer count results saved to {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['num_layers'], results_df['accuracy'], 'o-', linewidth=2)
    plt.title('Number of Layers vs Accuracy')
    plt.xlabel('Number of RNN Layers')
    plt.ylabel('Test Accuracy')
    plt.xticks(results_df['num_layers'])
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['num_layers'], results_df['f1_score'], 'o-', linewidth=2, color='orange')
    plt.title('Number of Layers vs F1 Score')
    plt.xlabel('Number of RNN Layers')
    plt.ylabel('Test F1 Score')
    plt.xticks(results_df['num_layers'])
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['num_layers'], results_df['loss'], 'o-', linewidth=2, color='red')
    plt.title('Number of Layers vs Loss')
    plt.xlabel('Number of RNN Layers')
    plt.ylabel('Test Loss')
    plt.xticks(results_df['num_layers'])
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_df['num_layers'], results_df['train_time'], 'o-', linewidth=2, color='green')
    plt.title('Number of Layers vs Training Time')
    plt.xlabel('Number of RNN Layers')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(results_df['num_layers'])
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = f"{PLOTS_DIR}/layer_count_experiment_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Layer count plot saved to {plot_path}")
    plt.close()
    
    return results_df

def run_unit_count_experiment():
    """Run experiment to test different numbers of RNN units"""
    print("\n" + "="*50)
    print("EXPERIMENT: RNN UNIT COUNT VARIATION")
    print("="*50)
    
    results = []
    
    # For each unit count in our variations
    for num_units in config.RNN_UNITS_VARIATIONS:
        model_name = f"rnn_units_{num_units}"
        print(f"\nTesting model with {num_units} RNN units")
        
        # Train model with this variation
        start_time = time.time()
        _, _, _, _, _, metrics = train_and_evaluate_model(
            rnn_units=num_units,
            model_name=model_name
        )
        train_time = time.time() - start_time
        
        # Store results
        results.append({
            'num_units': num_units,
            'accuracy': metrics['test_accuracy'],
            'f1_score': metrics['test_f1'],
            'loss': metrics['test_loss'],
            'train_time': train_time
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = f"{RESULTS_DIR}/unit_count_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Unit count results saved to {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['num_units'], results_df['accuracy'], 'o-', linewidth=2)
    plt.title('Number of Units vs Accuracy')
    plt.xlabel('Number of RNN Units')
    plt.ylabel('Test Accuracy')
    plt.xticks(results_df['num_units'])
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['num_units'], results_df['f1_score'], 'o-', linewidth=2, color='orange')
    plt.title('Number of Units vs F1 Score')
    plt.xlabel('Number of RNN Units')
    plt.ylabel('Test F1 Score')
    plt.xticks(results_df['num_units'])
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['num_units'], results_df['loss'], 'o-', linewidth=2, color='red')
    plt.title('Number of Units vs Loss')
    plt.xlabel('Number of RNN Units')
    plt.ylabel('Test Loss')
    plt.xticks(results_df['num_units'])
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_df['num_units'], results_df['train_time'], 'o-', linewidth=2, color='green')
    plt.title('Number of Units vs Training Time')
    plt.xlabel('Number of RNN Units')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(results_df['num_units'])
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = f"{PLOTS_DIR}/unit_count_experiment_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Unit count plot saved to {plot_path}")
    plt.close()
    
    return results_df

def run_bidirectional_experiment():
    """Run experiment to test bidirectional vs unidirectional RNN"""
    print("\n" + "="*50)
    print("EXPERIMENT: BIDIRECTIONAL RNN VARIATION")
    print("="*50)
    
    results = []
    
    # For each bidirectional setting in our variations
    for is_bidirectional in config.BIDIRECTIONAL_VARIATIONS:
        direction_name = "bidirectional" if is_bidirectional else "unidirectional"
        model_name = f"rnn_{direction_name}"
        print(f"\nTesting model with {direction_name} RNN")
        
        # Train model with this variation
        start_time = time.time()
        _, _, _, _, _, metrics = train_and_evaluate_model(
            bidirectional=is_bidirectional,
            model_name=model_name
        )
        train_time = time.time() - start_time
        
        # Store results
        results.append({
            'is_bidirectional': is_bidirectional,
            'direction': direction_name,
            'accuracy': metrics['test_accuracy'],
            'f1_score': metrics['test_f1'],
            'loss': metrics['test_loss'],
            'train_time': train_time
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = f"{RESULTS_DIR}/bidirectional_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Bidirectional results saved to {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create bar charts for categorical comparison
    directions = results_df['direction'].tolist()
    
    plt.subplot(2, 2, 1)
    plt.bar(directions, results_df['accuracy'], color=['blue', 'darkblue'])
    plt.title('Directionality vs Accuracy')
    plt.xlabel('RNN Type')
    plt.ylabel('Test Accuracy')
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 2)
    plt.bar(directions, results_df['f1_score'], color=['orange', 'darkorange'])
    plt.title('Directionality vs F1 Score')
    plt.xlabel('RNN Type')
    plt.ylabel('Test F1 Score')
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 3)
    plt.bar(directions, results_df['loss'], color=['red', 'darkred'])
    plt.title('Directionality vs Loss')
    plt.xlabel('RNN Type')
    plt.ylabel('Test Loss')
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 4)
    plt.bar(directions, results_df['train_time'], color=['green', 'darkgreen'])
    plt.title('Directionality vs Training Time')
    plt.xlabel('RNN Type')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plot_path = f"{PLOTS_DIR}/bidirectional_experiment_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Bidirectional plot saved to {plot_path}")
    plt.close()
    
    return results_df

def analyze_all_results(layer_results, unit_results, bidirectional_results):
    """Generate a comprehensive analysis of all experiment results"""
    
    print("\n" + "="*50)
    print("COMPREHENSIVE ANALYSIS OF ALL VARIATIONS")
    print("="*50)
    
    # Create summary table
    summary = pd.DataFrame({
        'Variation': [
            f"Layers: Best of {config.RNN_LAYERS_VARIATIONS}",
            f"Units: Best of {config.RNN_UNITS_VARIATIONS}",
            f"Direction: Best of {['Unidirectional', 'Bidirectional']}"
        ],
        'Best Config': [
            f"{layer_results.loc[layer_results['f1_score'].idxmax(), 'num_layers']} layers",
            f"{unit_results.loc[unit_results['f1_score'].idxmax(), 'num_units']} units",
            f"{'Bidirectional' if bidirectional_results.loc[bidirectional_results['f1_score'].idxmax(), 'is_bidirectional'] else 'Unidirectional'}"
        ],
        'Accuracy': [
            f"{layer_results['accuracy'].max():.4f}",
            f"{unit_results['accuracy'].max():.4f}",
            f"{bidirectional_results['accuracy'].max():.4f}"
        ],
        'F1 Score': [
            f"{layer_results['f1_score'].max():.4f}",
            f"{unit_results['f1_score'].max():.4f}",
            f"{bidirectional_results['f1_score'].max():.4f}"
        ],
        'Relative Impact': [
            f"{(layer_results['f1_score'].max() - layer_results['f1_score'].min()):.4f}",
            f"{(unit_results['f1_score'].max() - unit_results['f1_score'].min()):.4f}",
            f"{(bidirectional_results['f1_score'].max() - bidirectional_results['f1_score'].min()):.4f}"
        ]
    })
    
    # Save summary to CSV
    summary_path = f"{RESULTS_DIR}/variation_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Comprehensive summary saved to {summary_path}")
    
    # Print analysis
    print("\nSummary of Variation Impacts:")
    print(summary)
    
    # Determine which variation had the most impact
    impacts = [
        layer_results['f1_score'].max() - layer_results['f1_score'].min(),
        unit_results['f1_score'].max() - unit_results['f1_score'].min(),
        bidirectional_results['f1_score'].max() - bidirectional_results['f1_score'].min()
    ]
    
    variation_names = ['Number of Layers', 'Number of Units', 'Bidirectionality']
    most_impactful = variation_names[np.argmax(impacts)]
    
    print(f"\nMost impactful variation: {most_impactful}")
    print(f"Impact range: {max(impacts):.4f} F1 score difference")
    
    # Create a consolidated plot for best configurations
    plt.figure(figsize=(10, 6))
    
    variations = ['Layers', 'Units', 'Direction']
    best_configs = [
        f"{layer_results.loc[layer_results['f1_score'].idxmax(), 'num_layers']}",
        f"{unit_results.loc[unit_results['f1_score'].idxmax(), 'num_units']}",
        f"{'Bi' if bidirectional_results.loc[bidirectional_results['f1_score'].idxmax(), 'is_bidirectional'] else 'Uni'}"
    ]
    
    best_accuracies = [
        layer_results['accuracy'].max(),
        unit_results['accuracy'].max(),
        bidirectional_results['accuracy'].max()
    ]
    
    best_f1s = [
        layer_results['f1_score'].max(),
        unit_results['f1_score'].max(),
        bidirectional_results['f1_score'].max()
    ]
    
    # Create labels for x-axis showing best configuration
    x_labels = [f"{v}\n({c})" for v, c in zip(variations, best_configs)]
    
    # Plot
    x = np.arange(len(variations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, best_accuracies, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, best_f1s, width, label='F1 Score')
    
    ax.set_title('Best Configurations for Each Variation')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Save the plot
    plot_path = f"{PLOTS_DIR}/best_configurations_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Best configurations plot saved to {plot_path}")
    plt.close()
    
    # Return the optimal configuration based on F1 score
    optimal_config = {
        'num_rnn_layers': int(layer_results.loc[layer_results['f1_score'].idxmax(), 'num_layers']),
        'rnn_units': int(unit_results.loc[unit_results['f1_score'].idxmax(), 'num_units']),
        'bidirectional': bool(bidirectional_results.loc[bidirectional_results['f1_score'].idxmax(), 'is_bidirectional'])
    }
    
    print("\nOptimal configuration based on F1 score:")
    print(f"- Number of RNN layers: {optimal_config['num_rnn_layers']}")
    print(f"- Number of RNN units: {optimal_config['rnn_units']}")
    print(f"- Bidirectional: {optimal_config['bidirectional']}")
    
    return optimal_config

def run_optimal_model(optimal_config):
    """Train a model with the optimal configuration determined from experiments"""
    print("\n" + "="*50)
    print("TRAINING OPTIMAL MODEL CONFIGURATION")
    print("="*50)
    
    model_name = f"optimal_rnn_l{optimal_config['num_rnn_layers']}_u{optimal_config['rnn_units']}_{'bi' if optimal_config['bidirectional'] else 'uni'}"
    
    print(f"\nTraining optimal model: {model_name}")
    print(f"Configuration: {optimal_config}")
    
    # Train with optimal configuration
    model_results = train_and_evaluate_model(
        num_rnn_layers=optimal_config['num_rnn_layers'],
        rnn_units=optimal_config['rnn_units'],
        bidirectional=optimal_config['bidirectional'],
        model_name=model_name
    )
    
    model, history, _, _, _, metrics = model_results
    
    print("\nOptimal Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curves (Optimal Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Accuracy Curves (Optimal Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plot_path = f"{PLOTS_DIR}/optimal_model_training_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Optimal model training plots saved to {plot_path}")
    plt.close()
    
    return model, metrics

if __name__ == "__main__":
    print("Starting SimpleRNN variation experiments...")
    print(f"Results will be saved to {RESULTS_DIR} directory")
    
    # Run all experiments
    layer_results = run_layer_count_experiment()
    unit_results = run_unit_count_experiment()
    bidirectional_results = run_bidirectional_experiment()
    
    # Analyze all results together
    optimal_config = analyze_all_results(layer_results, unit_results, bidirectional_results)
    
    # Train the optimal model
    optimal_model, optimal_metrics = run_optimal_model(optimal_config)
    
    print("\nAll experiments completed!")
    print(f"Check {RESULTS_DIR} for detailed results and {PLOTS_DIR} for visualizations")