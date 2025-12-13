"""
Visualization utilities for Snake RL Experiments
Generates training curves, comparison plots, and summary charts
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = './results'
FIGURES_DIR = './figures'

# Color palette for consistent visualization
COLORS = {
    'baseline': '#2ecc71',
    'wide': '#3498db', 
    'deep_4hidden': '#9b59b6',
    'deep_6hidden': '#e74c3c',
    '4 hidden': '#9b59b6',
    '6 hidden': '#e74c3c',
    'small': '#f39c12',
    'large': '#1abc9c',
    'small_memory': '#f39c12',
    'large_memory': '#1abc9c',
    'wall': '#e67e22',
}


def load_results_csv(filename):
    """Load experiment results from CSV file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    results = {
        'metadata': {},
        'episodes': [],
        'scores': [],
        'records': [],
        'avg_100': [],
    }
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        in_data = False
        
        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                # Parse metadata
                key = row[0][2:].strip()
                value = row[1] if len(row) > 1 else ''
                results['metadata'][key] = value
            elif row[0] == 'Episode':
                in_data = True
                continue
            elif in_data:
                results['episodes'].append(int(row[0]))
                results['scores'].append(int(row[1]))
                results['records'].append(int(row[2]))
                results['avg_100'].append(float(row[3]))
    
    return results


def smooth_curve(values, window=10):
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    return smoothed


def plot_training_curve(results, title, filename, show_smoothed=True):
    """Plot training curve for a single experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = results['episodes']
    scores = results['scores']
    avg_100 = results['avg_100']
    
    # Plot raw scores with low alpha
    ax.plot(episodes, scores, alpha=0.3, color='blue', linewidth=0.5, label='Score per Episode')
    
    # Plot rolling average
    ax.plot(episodes, avg_100, color='red', linewidth=2, label='Rolling Avg (100 eps)')
    
    # Optionally add smoothed curve
    if show_smoothed:
        smoothed = smooth_curve(scores, window=20)
        ax.plot(episodes, smoothed, color='green', linewidth=1.5, alpha=0.8, 
                label='Smoothed (20-ep window)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(episodes))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison(results_dict, title, filename, metric='avg_100'):
    """Plot comparison of multiple experiments."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for label, results in results_dict.items():
        if results is None:
            continue
        episodes = results['episodes']
        values = results[metric]
        
        # Determine color
        color = 'gray'
        for key in COLORS:
            if key in label.lower():
                color = COLORS[key]
                break
        
        ax.plot(episodes, values, linewidth=2, label=label, color=color)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Rolling Average Score (100 episodes)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_bar_comparison(summary_data, title, filename):
    """Create bar chart comparing final performance across experiments."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    experiments = list(summary_data.keys())
    final_avgs = [summary_data[exp]['final_avg'] for exp in experiments]
    std_devs = [summary_data[exp]['std_dev'] for exp in experiments]
    
    # Determine colors based on experiment type
    colors = []
    for exp in experiments:
        color = '#95a5a6'  # default gray
        for key in COLORS:
            if key in exp.lower():
                color = COLORS[key]
                break
        colors.append(color)
    
    x = np.arange(len(experiments))
    bars = ax.bar(x, final_avgs, yerr=std_devs, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Average Score (Last 100 Episodes)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_avgs):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_stability_analysis(results_dict, filename):
    """Plot training stability (variance over time) for observational analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Variance over time
    ax1 = axes[0]
    for label, results in results_dict.items():
        if results is None:
            continue
        episodes = results['episodes']
        # Load variance data from episode data if available
        # For now, use rolling std as proxy
        scores = results['scores']
        rolling_std = []
        for i in range(len(scores)):
            start = max(0, i - 50)
            rolling_std.append(np.std(scores[start:i+1]))
        
        color = 'gray'
        for key in COLORS:
            if key in label.lower():
                color = COLORS[key]
                break
        ax1.plot(episodes, rolling_std, linewidth=1.5, label=label, color=color, alpha=0.8)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Rolling Std Dev (50 eps)', fontsize=12)
    ax1.set_title('Training Stability: Score Variance Over Time', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Performance drop from peak (forgetting indicator)
    ax2 = axes[1]
    for label, results in results_dict.items():
        if results is None:
            continue
        episodes = results['episodes']
        scores = results['scores']
        
        # Calculate cumulative peak and drop
        peak_so_far = []
        drops = []
        current_peak = 0
        for i, score in enumerate(scores):
            rolling_avg = np.mean(scores[max(0, i-100):i+1])
            if rolling_avg > current_peak:
                current_peak = rolling_avg
            peak_so_far.append(current_peak)
            drops.append(current_peak - rolling_avg)
        
        color = 'gray'
        for key in COLORS:
            if key in label.lower():
                color = COLORS[key]
                break
        ax2.plot(episodes, drops, linewidth=1.5, label=label, color=color, alpha=0.8)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Drop from Peak Avg', fontsize=12)
    ax2.set_title('Catastrophic Forgetting Indicator: Performance Drop from Peak', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def generate_all_visualizations():
    """Generate all visualizations from saved results."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50 + "\n")
    
    # Load all results
    all_results = {}
    
    # Part A
    baseline = load_results_csv('part_a_baseline.csv')
    if baseline:
        all_results['Baseline'] = baseline
        plot_training_curve(baseline, 'Part A: Baseline Training Curve (11→256→3)', 
                           'part_a_training_curve.png')
    
    # Part B - Architecture variations
    arch_results = {}
    arch_files = {
        'Baseline (1 hidden)': 'part_a_baseline.csv',
        'Wide (1 hidden, 512)': 'part_b_wide.csv',
        'Deep (4 hidden)': 'part_b_deep_4hidden.csv',
        'Deep (6 hidden)': 'part_b_deep_6hidden.csv',
    }
    
    for label, filename in arch_files.items():
        result = load_results_csv(filename)
        if result:
            arch_results[label] = result
    
    if arch_results:
        plot_comparison(arch_results, 'Part B: Architecture Comparison', 
                       'part_b_architecture_comparison.png')
    
    # Part C - Memory variations
    mem_results = {}
    mem_files = {
        'Baseline (100K)': 'part_a_baseline.csv',
        'Small Memory (10K)': 'part_c_memory_small.csv',
        'Large Memory (500K)': 'part_c_memory_large.csv',
    }
    
    for label, filename in mem_files.items():
        result = load_results_csv(filename)
        if result:
            mem_results[label] = result
    
    if mem_results:
        plot_comparison(mem_results, 'Part C: Memory Size Comparison',
                       'part_c_memory_comparison.png')
    
    # Part D - Wall experiments
    wall_results = {}
    wall_files = {
        'Wall + Baseline': 'd_a1_wall_baseline.csv',
        'Wall + Wide': 'd_b1_wall_wide.csv',
        'Wall + Deep 4-hidden': 'd_b2_wall_deep_4hidden.csv',
        'Wall + Deep 6-hidden': 'd_b3_wall_deep_6hidden.csv',
        'Wall + Small Memory': 'd_c1_wall_small_memory.csv',
        'Wall + Large Memory': 'd_c2_wall_large_memory.csv',
    }
    
    for label, filename in wall_files.items():
        result = load_results_csv(filename)
        if result:
            wall_results[label] = result
    
    if wall_results:
        plot_comparison(wall_results, 'Part D: Wall Environment Comparison',
                       'part_d_wall_comparison.png')
    
    # No-wall vs Wall comparison for baseline
    wall_comparison = {}
    if baseline:
        wall_comparison['Baseline (No Wall)'] = baseline
    wall_baseline = load_results_csv('d_a1_wall_baseline.csv')
    if wall_baseline:
        wall_comparison['Baseline (With Wall)'] = wall_baseline
    
    if len(wall_comparison) == 2:
        plot_comparison(wall_comparison, 'Environment Complexity: Wall Impact on Baseline',
                       'wall_impact_baseline.png')
    
    # Summary bar chart
    summary_data = {}
    for label, results in {**all_results, **arch_results, **mem_results, **wall_results}.items():
        if results:
            # Get final avg from last 100 episodes
            final_scores = results['scores'][-100:]
            summary_data[label] = {
                'final_avg': np.mean(final_scores),
                'std_dev': np.std(final_scores)
            }
    
    if summary_data:
        plot_bar_comparison(summary_data, 'Overall Performance Summary',
                           'summary_bar_chart.png')
    
    # Generate stability analysis plots
    if arch_results:
        plot_stability_analysis(arch_results, 'stability_analysis_architectures.png')
    
    if mem_results:
        plot_stability_analysis(mem_results, 'stability_analysis_memory.png')
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE")
    print("="*50 + "\n")


def create_latex_tables():
    """Generate LaTeX table code for the report."""
    print("\nGenerating LaTeX table code...")
    
    # Load summary results
    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    if not os.path.exists(summary_file):
        print("Summary file not found. Run experiments first.")
        return
    
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    latex = """
\\begin{table}[h]
\\centering
\\caption{Summary of Experiment Results}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Experiment} & \\textbf{Architecture} & \\textbf{Memory} & \\textbf{Wall} & \\textbf{Final Avg} & \\textbf{Record} \\\\
\\hline
"""
    
    for row in rows:
        latex += f"{row['Experiment']} & {row['Architecture']} & {row['Memory Size']} & "
        latex += f"{row['Wall']} & {row['Final Avg (100)']} & {row['Final Record']} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    
    print(latex)


if __name__ == '__main__':
    generate_all_visualizations()
    create_latex_tables()
