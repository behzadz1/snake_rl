# Snake RL Experiment Suite

## Overview
This repository contains a comprehensive reinforcement learning experiment suite for training a DQN agent to play Snake. The experiments explore the effects of neural network architecture, replay memory size, and environment complexity on learning performance.

## Requirements
- Python 3.8+ (tested on Python 3.10, 3.11)
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Mac Installation (Step-by-Step)

### 1. Open Terminal and navigate to project folder
```bash
cd ~/Desktop/snake_rl_experiments
```

### 2. Create virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install torch numpy matplotlib
```

### For Apple Silicon (M1/M2/M3) Macs:
```bash
pip install torch numpy matplotlib
# PyTorch automatically detects Apple Silicon and uses MPS acceleration
```

## Project Structure
```
snake_rl/
├── baseline/                    # Original unmodified files
│   ├── agent.py
│   ├── game.py
│   ├── helper.py
│   ├── model.py
│   └── snake_game_human.py
├── experiment_driver.py         # Main experiment orchestration script
├── game_headless.py             # Headless game for fast training
├── agent_configurable.py        # Configurable DQN agent
├── model_configurable.py        # Configurable neural network
├── visualizations.py            # Plotting and visualization utilities
├── results/                     # CSV output files
└── figures/                     # Generated plots
```

## Running Experiments

### RECOMMENDED: Run All Experiments (Parts A-D)
```bash
python experiment_driver.py --all
```

### Run Specific Parts (if you want to run individually)
```bash
python experiment_driver.py --part a    # Part A: Baseline (~3 min)
python experiment_driver.py --part b    # Part B: Architecture variations (~10 min)
python experiment_driver.py --part c    # Part C: Memory size variations (~7 min)
python experiment_driver.py --part d    # Part D: Wall obstacle experiments (~15 min)
```

### Quick Test Run (to verify setup works)
```bash
python experiment_driver.py --all --episodes 50   # ~5 min total
```

### Custom Episode Count
```bash
python experiment_driver.py --all --episodes 200   # Faster run for testing
```

## Estimated Run Times (500 episodes each, M1 Mac)
| Part | Experiments | Est. Time |
|------|-------------|-----------|
| A | 1 baseline | ~3 min |
| B | 3 architectures | ~10 min |
| C | 2 memory configs | ~7 min |
| D | 6 wall experiments | ~18 min |
| **Total** | **12 experiments** | **~40-50 min** |

*Times may vary based on your Mac's specs. Intel Macs may be 20-30% slower.*

## Experiment Configuration

### Part A: Baseline
- Architecture: 11 → 256 → 3 (standard 2-layer network)
- Memory: 100,000 experiences
- Environment: Standard (no obstacles)
- Episodes: 500

### Part B: Architecture Variations
| Variant | Architecture | Hidden Layers |
|---------|--------------|---------------|
| Wide | 11 → 512 → 3 | 1 (wider) |
| Deep 4-hidden | 11 → 256 → 256 → 128 → 64 → 3 | 4 |
| Deep 6-hidden | 11 → 256 → 256 → 256 → 128 → 128 → 64 → 3 | 6 |

### Part C: Memory Size Variations
| Variant | Buffer Size | Trade-off |
|---------|-------------|-----------|
| Small | 10,000 | Faster but potentially unstable |
| Large | 500,000 | More stable but slower convergence |

### Part D: Environment Complexity
All Part A, B, and C configurations repeated with a static wall obstacle added to the environment.

## Output Files

### Results (CSV)
- `part_a_baseline.csv` - Baseline results
- `part_b_*.csv` - Architecture experiment results
- `part_c_*.csv` - Memory experiment results
- `d_*.csv` - Wall experiment results
- `experiment_summary.csv` - Consolidated summary

**CSV Columns include:**
- Episode, Score, Record, Avg_Last_100, Steps
- Memory_Usage (replay buffer fill level)
- Variance_50 (training stability - variance over 50 eps)
- Std_50 (standard deviation over 50 eps)
- Peak_Avg (best rolling average achieved)
- Performance_Drop (drop from peak - forgetting indicator)
- Epsilon (exploration rate)

### Figures (PNG)
- `part_a_training_curve.png` - Baseline training curve
- `part_b_architecture_comparison.png` - Architecture comparison
- `part_c_memory_comparison.png` - Memory size comparison
- `part_d_wall_comparison.png` - Wall environment results
- `summary_bar_chart.png` - Overall performance comparison
- `stability_analysis_architectures.png` - Stability/forgetting analysis
- `stability_analysis_memory.png` - Memory config stability analysis

## Generating Visualizations
After running experiments:
```bash
python visualizations.py
```


## Complete Workflow Summary
```bash
# Step 1: Setup
cd ~/Desktop/snake_rl_experiments
python3 -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib

# Step 2: Run experiments (35-45 min)
python experiment_driver.py --all

# Step 3: Generate figures
python visualizations.py

```

## Notes
- The headless game version runs without pygame display for ~10x faster training
- Each experiment with 500 episodes takes approximately 2-5 minutes depending on configuration
- Full suite (11 experiments) takes approximately 30-60 minutes
- Results are saved incrementally - can resume if interrupted

## Academic References
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Lin, L.J. (1992). Self-improving reactive agents based on reinforcement learning. Machine Learning.
- Schaul, T., et al. (2015). Prioritized Experience Replay. ICLR.
