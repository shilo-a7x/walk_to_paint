"""Extract and analyze training results from logs"""
import re
import os

datasets = {
    'wiki-rfa': ('outputs/wiki-rfa/wiki-rfa-run_20260208-145124', 'nohup_logs/wiki-rfa_20260208-145121.log'),
    'epinions': ('outputs/epinions/epinions-run_20260208-145126', 'nohup_logs/epinions_20260208-145121.log'),
    'slashdot090221': ('outputs/slashdot090221/slashdot090221-run_20260208-145128', 'nohup_logs/slashdot090221_20260208-145121.log'),
}

for dataset_name, (exp_dir, log_file) in datasets.items():
    print(f"\n{'='*70}")
    print(f"{dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Extract metrics from log
    if os.path.exists(log_file):
        # Find all epoch metrics lines
        cmd = f"grep -o 'Epoch [0-9]*:.*val_auc_epoch=[0-9.]*' {log_file} | tail -30"
        os.system(cmd)
        
        # Find early stopping message
        cmd = f"grep 'Best score\\|did not improve' {log_file} | tail -2"
        print("\nEarly Stopping Info:")
        os.system(cmd)
        
        # Find final test results
        cmd = f"grep -A 10 'Test metric.*DataLoader' {log_file} | head -15"
        print("\nFinal Test Results:")
        os.system(cmd)
    
    # List saved predictions
    pred_dir = os.path.join(exp_dir, 'checkpoints', f'{dataset_name}_predictions')
    if os.path.exists(pred_dir):
        epochs = sorted([d for d in os.listdir(pred_dir) if d.startswith('epoch_')])
        print(f"\nPredictions saved for epochs: {epochs[0]} to {epochs[-1]} ({len(epochs)} epochs)")

