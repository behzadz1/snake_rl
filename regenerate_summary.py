
import os
import csv
import glob
import numpy as np

RESULTS_DIR = './results'

def load_results_csv(filepath):
    """Load experiment results from CSV file."""
    results = {
        'metadata': {},
        'episodes': [],
        'scores': [],
        'avg_100': [],
        'final_record': 0,
        'final_avg_100': 0,
        'overall_avg': 0,
        'block_averages': [],
        'avg_final_variance': 'N/A',
        'max_performance_drop': 'N/A',
        'exploration_end_episode': 'N/A'
    }
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        in_data = False
        
        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                key = row[0][2:].strip()
                if '---' in key or 'Results' in key: continue
                value = row[1] if len(row) > 1 else ''
                results['metadata'][key] = value
            elif row[0] == 'Episode':
                in_data = True
                continue
            elif in_data:
                results['scores'].append(float(row[1]))
                results['avg_100'].append(float(row[3]))

    # Extract metadata values with fallbacks
    meta = results['metadata']
    results['experiment_name'] = meta.get('Name', 'Unknown')
    results['arch_string'] = meta.get('Architecture', 'Unknown')
    results['memory_size'] = meta.get('Memory Size', 'Unknown')
    results['with_wall'] = meta.get('Wall Obstacle', 'Unknown')
    results['final_record'] = meta.get('Final Record', '0')
    results['final_avg_100'] = meta.get('Final Avg (last 100)', '0')
    results['overall_avg'] = meta.get('Overall Avg', '0')
    
    # Parse block averages if present (it's a single string in the CSV usually?)
    # Based on experiment_driver.py: 
    # writer.writerow(['# Block Averages (per 100 eps)', ', '.join(...)])
    block_avgs = meta.get('Block Averages (per 100 eps)', '')
    if block_avgs:
         results['block_averages'] = [b.strip() for b in block_avgs.split(',')]
    
    results['avg_final_variance'] = meta.get('Training Stability (avg variance)', 'N/A')
    results['max_performance_drop'] = meta.get('Max Performance Drop (forgetting)', 'N/A')
    results['exploration_end_episode'] = meta.get('Exploration Ended at Episode', 'N/A')
    
    return results

def main():
    print("Regenerating experiment_summary.csv...")
    csv_files = glob.glob(os.path.join(RESULTS_DIR, '*.csv'))
    summary_rows = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        if filename == 'experiment_summary.csv':
            continue
            
        print(f"Processing {filename}...")
        try:
            res = load_results_csv(filepath)
            
            row = [
                res['experiment_name'],
                res['arch_string'],
                res['memory_size'],
                res['with_wall'],
                res['final_record'],
                res['final_avg_100'],
                res['overall_avg'],
                'N/A', # Std Dev not easily parsed from metadata unless we recalculate
            ]
            
            # Add block averages (pad to 5 blocks)
            blocks = res.get('block_averages', [])
            for i in range(5):
                if i < len(blocks):
                    row.append(blocks[i])
                else:
                    row.append('N/A')
            
            row.append(res['avg_final_variance'])
            row.append(res['max_performance_drop'])
            row.append(res['exploration_end_episode'])
            row.append('N/A') # Time
            
            summary_rows.append(row)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Sort rows by experiment name
    summary_rows.sort(key=lambda x: x[0])

    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment', 'Architecture', 'Memory Size', 'Wall',
            'Final Record', 'Final Avg (100)', 'Overall Avg', 'Std Dev',
            'Block 1 Avg', 'Block 2 Avg', 'Block 3 Avg', 'Block 4 Avg', 'Block 5 Avg',
            'Stability (Var)', 'Max Perf Drop', 'Explore End Ep',
            'Time (s)'
        ])
        writer.writerows(summary_rows)
        
    print(f"Summary saved to {summary_file}")

if __name__ == '__main__':
    main()
