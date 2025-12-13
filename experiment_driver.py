"""
Snake RL Experiment Driver
Runs all experiments for Parts A-D and logs results

Usage:
    python experiment_driver.py --all           # Run all experiments
    python experiment_driver.py --part a        # Run only Part A
    python experiment_driver.py --part b        # Run only Part B
    python experiment_driver.py --part c        # Run only Part C
    python experiment_driver.py --part d        # Run only Part D
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

# Import experiment components
from game_headless import SnakeGameHeadless
from agent_configurable import ConfigurableAgent
from model_configurable import ARCHITECTURES

# Experiment configuration
NUM_EPISODES = 500
RESULTS_DIR = './results'
FIGURES_DIR = './figures'

# Memory configurations for Part C
MEMORY_CONFIGS = {
    'small': 10000,      # Small buffer - faster but less stable
    'baseline': 100000,  # Standard buffer
    'large': 500000,     # Large buffer - more stable but slower
}


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def run_experiment(experiment_name, architecture='baseline', memory_size=100000, 
                   with_wall=False, num_episodes=None, verbose=True):
    """
    Run a single experiment configuration.
    
    Args:
        experiment_name: Identifier for this experiment
        architecture: Architecture key or layer list
        memory_size: Replay buffer size
        with_wall: Whether to include wall obstacle
        with_wall: Whether to include wall obstacle
        num_episodes: Number of training episodes
        verbose: Print progress updates
    
    Returns:
        dict: Results including scores, averages, and metadata
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"Architecture: {architecture}")
        print(f"Memory Size: {memory_size:,}")
        print(f"Wall Obstacle: {with_wall}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}\n")
    
    if num_episodes is None:
        num_episodes = NUM_EPISODES
    
    # Initialize agent and game
    agent = ConfigurableAgent(architecture=architecture, memory_size=memory_size)
    game = SnakeGameHeadless(with_wall=with_wall)
    
    # Tracking variables
    scores = []
    episode_data = []
    start_time = time.time()
    record = 0
    
    for episode in range(num_episodes):
        # Reset game for new episode
        game.reset()
        episode_score = 0
        steps = 0
        
        while True:
            # Get state and action
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            
            # Execute action
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)
            
            # Train short memory
            agent.train_short_memory(state_old, action, reward, state_new, done)
            
            # Store experience
            agent.remember(state_old, action, reward, state_new, done)
            
            steps += 1
            
            if done:
                # End of episode
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                episode_score = score
                if score > record:
                    record = score
                
                break
        
        scores.append(episode_score)
        
        # Calculate running averages
        avg_last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Calculate stability metrics (variance in recent window)
        stability_window = 50
        if len(scores) >= stability_window:
            recent_variance = np.var(scores[-stability_window:])
            recent_std = np.std(scores[-stability_window:])
        else:
            recent_variance = np.var(scores) if len(scores) > 1 else 0
            recent_std = np.std(scores) if len(scores) > 1 else 0
        
        # Track catastrophic forgetting (drop from peak performance)
        peak_avg = max([np.mean(scores[max(0,i-100):i+1]) for i in range(len(scores))])
        performance_drop = peak_avg - avg_last_100 if peak_avg > 0 else 0
        
        # Track exploration (current epsilon value)
        current_epsilon = max(0, 80 - agent.n_games)
        
        # Store episode data
        episode_data.append({
            'episode': episode + 1,
            'score': episode_score,
            'record': record,
            'avg_last_100': avg_last_100,
            'steps': steps,
            'memory_usage': len(agent.memory),
            'variance_50': recent_variance,
            'std_50': recent_std,
            'peak_avg': peak_avg,
            'performance_drop': performance_drop,
            'epsilon': current_epsilon
        })
        
        # Progress update every 50 episodes
        if verbose and (episode + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta = (num_episodes - episode - 1) / eps_per_sec
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Score: {episode_score} | Record: {record} | "
                  f"Avg(100): {avg_last_100:.2f} | "
                  f"ETA: {eta:.0f}s")
    
    # Calculate summary statistics
    elapsed_time = time.time() - start_time
    
    # Calculate averages for each 100-episode block
    block_averages = []
    for i in range(0, num_episodes, 100):
        block = scores[i:i+100]
        block_averages.append(np.mean(block))
    
    # Calculate observational summary metrics
    final_episode_data = episode_data[-1] if episode_data else {}
    
    # Stability: average variance over last 100 episodes
    last_100_variances = [ep['variance_50'] for ep in episode_data[-100:]]
    avg_final_variance = np.mean(last_100_variances) if last_100_variances else 0
    
    # Catastrophic forgetting: max performance drop observed
    max_performance_drop = max([ep['performance_drop'] for ep in episode_data]) if episode_data else 0
    
    # Exploration: when did exploration effectively end (epsilon < 5)
    exploration_end_episode = None
    for ep in episode_data:
        if ep['epsilon'] <= 5:
            exploration_end_episode = ep['episode']
            break
    
    results = {
        'experiment_name': experiment_name,
        'architecture': str(architecture),
        'memory_size': memory_size,
        'with_wall': with_wall,
        'num_episodes': num_episodes,
        'scores': scores,
        'episode_data': episode_data,
        'final_record': record,
        'final_avg_100': np.mean(scores[-100:]),
        'overall_avg': np.mean(scores),
        'overall_std': np.std(scores),
        'block_averages': block_averages,
        'elapsed_time': elapsed_time,
        'arch_string': agent.model.get_architecture_string(),
        # Observational metrics
        'avg_final_variance': avg_final_variance,
        'max_performance_drop': max_performance_drop,
        'exploration_end_episode': exploration_end_episode
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {experiment_name}")
        print(f"Final Record: {record}")
        print(f"Final Avg (last 100): {results['final_avg_100']:.2f}")
        print(f"Overall Avg: {results['overall_avg']:.2f} ± {results['overall_std']:.2f}")
        print(f"100-Episode Block Averages: {[f'{x:.2f}' for x in block_averages]}")
        print(f"--- Observational Notes ---")
        print(f"Training Stability (avg variance): {results['avg_final_variance']:.2f}")
        print(f"Max Performance Drop (forgetting): {results['max_performance_drop']:.2f}")
        print(f"Exploration Ended at Episode: {results['exploration_end_episode']}")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        print(f"{'='*60}\n")
    
    return results


def save_results_csv(results, filename):
    """Save experiment results to CSV file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata header
        writer.writerow(['# Experiment Results'])
        writer.writerow(['# Name', results['experiment_name']])
        writer.writerow(['# Architecture', results['arch_string']])
        writer.writerow(['# Memory Size', results['memory_size']])
        writer.writerow(['# Wall Obstacle', results['with_wall']])
        writer.writerow(['# Final Record', results['final_record']])
        writer.writerow(['# Final Avg (last 100)', f"{results['final_avg_100']:.2f}"])
        writer.writerow(['# Overall Avg', f"{results['overall_avg']:.2f}"])
        writer.writerow(['# Overall Std', f"{results['overall_std']:.2f}"])
        writer.writerow(['# Block Averages (per 100 eps)', 
                        ', '.join([f'{x:.2f}' for x in results['block_averages']])])
        writer.writerow(['# --- Observational Notes ---'])
        writer.writerow(['# Training Stability (avg variance)', f"{results['avg_final_variance']:.2f}"])
        writer.writerow(['# Max Performance Drop (forgetting)', f"{results['max_performance_drop']:.2f}"])
        writer.writerow(['# Exploration Ended at Episode', results['exploration_end_episode']])
        writer.writerow([])
        
        # Write episode data
        writer.writerow(['Episode', 'Score', 'Record', 'Avg_Last_100', 'Steps', 'Memory_Usage',
                        'Variance_50', 'Std_50', 'Peak_Avg', 'Performance_Drop', 'Epsilon'])
        for ep_data in results['episode_data']:
            writer.writerow([
                ep_data['episode'],
                ep_data['score'],
                ep_data['record'],
                f"{ep_data['avg_last_100']:.2f}",
                ep_data['steps'],
                ep_data['memory_usage'],
                f"{ep_data['variance_50']:.2f}",
                f"{ep_data['std_50']:.2f}",
                f"{ep_data['peak_avg']:.2f}",
                f"{ep_data['performance_drop']:.2f}",
                ep_data['epsilon']
            ])
    
    print(f"Results saved to: {filepath}")


def run_part_a():
    """Part A: Baseline experiment with standard configuration."""
    print("\n" + "="*70)
    print("PART A: BASELINE EXPERIMENT")
    print("="*70)
    
    results = run_experiment(
        experiment_name='A1_baseline',
        architecture='baseline',  # 11 → 256 → 3
        memory_size=MEMORY_CONFIGS['baseline'],
        with_wall=False
    )
    
    save_results_csv(results, 'part_a_baseline.csv')
    return {'A1_baseline': results}


def run_part_b():
    """Part B: Neural network architecture variations."""
    print("\n" + "="*70)
    print("PART B: NEURAL NETWORK ARCHITECTURE EXPERIMENTS")
    print("="*70)
    
    experiments = {
        'B1_wide': 'wide',                 # 11 → 512 → 3 (1 hidden, wider)
        'B2_deep_4hidden': 'deep_4hidden', # 11 → 256 → 256 → 128 → 64 → 3 (4 hidden)
        'B3_deep_6hidden': 'deep_6hidden', # 11 → 256 → 256 → 256 → 128 → 128 → 64 → 3 (6 hidden)
    }
    
    all_results = {}
    
    for exp_name, arch in experiments.items():
        results = run_experiment(
            experiment_name=exp_name,
            architecture=arch,
            memory_size=MEMORY_CONFIGS['baseline'],
            with_wall=False
        )
        save_results_csv(results, f'part_b_{arch}.csv')
        all_results[exp_name] = results
    
    return all_results


def run_part_c():
    """Part C: Replay memory size variations."""
    print("\n" + "="*70)
    print("PART C: REPLAY MEMORY SIZE EXPERIMENTS")
    print("="*70)
    
    experiments = {
        'C1_small_memory': ('baseline', MEMORY_CONFIGS['small']),
        'C2_large_memory': ('baseline', MEMORY_CONFIGS['large']),
    }
    
    all_results = {}
    
    for exp_name, (arch, mem_size) in experiments.items():
        results = run_experiment(
            experiment_name=exp_name,
            architecture=arch,
            memory_size=mem_size,
            with_wall=False
        )
        mem_label = 'small' if mem_size == MEMORY_CONFIGS['small'] else 'large'
        save_results_csv(results, f'part_c_memory_{mem_label}.csv')
        all_results[exp_name] = results
    
    return all_results


def run_part_d():
    """Part D: Environment complexity with wall obstacle."""
    print("\n" + "="*70)
    print("PART D: ENVIRONMENT COMPLEXITY EXPERIMENTS (WITH WALL)")
    print("="*70)
    
    experiments = [
        # Baseline with wall
        ('D_A1_wall_baseline', 'baseline', MEMORY_CONFIGS['baseline']),
        
        # ALL Part B architecture variants with wall
        ('D_B1_wall_wide', 'wide', MEMORY_CONFIGS['baseline']),
        ('D_B2_wall_deep_4hidden', 'deep_4hidden', MEMORY_CONFIGS['baseline']),
        ('D_B3_wall_deep_6hidden', 'deep_6hidden', MEMORY_CONFIGS['baseline']),
        
        # ALL Part C memory variants with wall
        ('D_C1_wall_small_memory', 'baseline', MEMORY_CONFIGS['small']),
        ('D_C2_wall_large_memory', 'baseline', MEMORY_CONFIGS['large']),
    ]
    
    all_results = {}
    
    for exp_name, arch, mem_size in experiments:
        results = run_experiment(
            experiment_name=exp_name,
            architecture=arch,
            memory_size=mem_size,
            with_wall=True
        )
        save_results_csv(results, f'{exp_name.lower()}.csv')
        all_results[exp_name] = results
    
    return all_results


def generate_summary_report(all_results):
    """Generate summary CSV with all experiment results."""
    filepath = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment', 'Architecture', 'Memory Size', 'Wall',
            'Final Record', 'Final Avg (100)', 'Overall Avg', 'Std Dev',
            'Block 1 Avg', 'Block 2 Avg', 'Block 3 Avg', 'Block 4 Avg', 'Block 5 Avg',
            'Stability (Var)', 'Max Perf Drop', 'Explore End Ep',
            'Time (s)'
        ])
        
        for exp_name, results in all_results.items():
            row = [
                exp_name,
                results['arch_string'],
                results['memory_size'],
                'Yes' if results['with_wall'] else 'No',
                results['final_record'],
                f"{results['final_avg_100']:.2f}",
                f"{results['overall_avg']:.2f}",
                f"{results['overall_std']:.2f}",
            ]
            # Add block averages (pad with N/A if fewer than 5 blocks)
            for i in range(5):
                if i < len(results['block_averages']):
                    row.append(f"{results['block_averages'][i]:.2f}")
                else:
                    row.append('N/A')
            # Add observational metrics
            row.append(f"{results['avg_final_variance']:.2f}")
            row.append(f"{results['max_performance_drop']:.2f}")
            row.append(results['exploration_end_episode'])
            row.append(f"{results['elapsed_time']:.1f}")
            writer.writerow(row)
    
    print(f"\nSummary report saved to: {filepath}")


def main():
    global NUM_EPISODES
    parser = argparse.ArgumentParser(description='Snake RL Experiment Driver')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--part', type=str, choices=['a', 'b', 'c', 'd'], 
                       help='Run specific part')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help=f'Number of episodes per experiment (default: {NUM_EPISODES})')
    
    args = parser.parse_args()
    
    # Update global episodes if specified
    # Update global episodes if specified
    NUM_EPISODES = args.episodes
    
    ensure_directories()
    
    all_results = {}
    
    print("\n" + "="*70)
    print("SNAKE RL EXPERIMENT SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Episodes per experiment: {NUM_EPISODES}")
    print("="*70)
    
    start_time = time.time()
    
    if args.all or args.part == 'a':
        all_results.update(run_part_a())
    
    if args.all or args.part == 'b':
        all_results.update(run_part_b())
    
    if args.all or args.part == 'c':
        all_results.update(run_part_c())
    
    if args.all or args.part == 'd':
        all_results.update(run_part_d())
    
    if all_results:
        generate_summary_report(all_results)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    # If no arguments, run all experiments
    if len(sys.argv) == 1:
        sys.argv.append('--all')
    main()
