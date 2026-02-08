import re
import matplotlib.pyplot as plt
import os
import shutil

LOGS_PATH = 'Model/logs'
PLOTS_PATH = 'Model/plots'

def clear_plots_folder(plots_dir=LOGS_PATH):
    #clear the plots folder, creating it if it doesn't exist
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)
    print(f"Cleared and created '{plots_dir}/' directory")

def parse_log_file(filepath):
    """
    Parse training log file and extract epoch data and hyperparameters.
    
    Args:
        filepath: Path to the text file containing training logs
        
    Returns:
        Tuple of (epochs, train_losses, train_accs, val_losses, val_accs, hyperparams)
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract hyperparameters
    hyperparams = {}
    batch_size_match = re.search(r'BATCH_SIZE\s*=\s*(\d+)', content)
    epochs_match = re.search(r'EPOCHS\s*=\s*(\d+)', content)
    lr_match = re.search(r'LR\s*=\s*([\d.e\-+]+)', content)
    
    if batch_size_match:
        hyperparams['batch_size'] = int(batch_size_match.group(1))
    if epochs_match:
        hyperparams['epochs'] = int(epochs_match.group(1))
    if lr_match:
        hyperparams['lr'] = lr_match.group(1)
    
    # Extract epoch data using regex
    pattern = r'Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+) Train Acc: ([\d.]+) Val Loss: ([\d.]+) Val Acc: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    # Parse data into lists
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for match in matches:
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
        train_accs.append(float(match[2]))
        val_losses.append(float(match[3]))
        val_accs.append(float(match[4]))
    
    return epochs, train_losses, train_accs, val_losses, val_accs, hyperparams

def create_plots(epochs, train_losses, train_accs, val_losses, val_accs, 
                 log_number, hyperparams, plots_dir=PLOTS_PATH):
    #hyperparameters
    hyperparam_text = []
    if 'batch_size' in hyperparams:
        hyperparam_text.append(f"Batch Size: {hyperparams['batch_size']}")
    if 'epochs' in hyperparams:
        hyperparam_text.append(f"Epochs: {hyperparams['epochs']}")
    if 'lr' in hyperparams:
        hyperparam_text.append(f"LR: {hyperparams['lr']}")
    hyperparam_str = '\n'.join(hyperparam_text)
    
    #plot 1 - loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss (Run {log_number})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    if hyperparam_text:
        plt.text(0.02, 0.98, hyperparam_str, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    loss_path = os.path.join(plots_dir, f'plot{log_number}_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    #plot 2 - accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Training and Validation Accuracy (Run {log_number})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.ylim([0.5, 1.0])  #start from 0.5 since random guessing is 0.5 for binary
    
    if hyperparam_text:
        plt.text(0.02, 0.98, hyperparam_str, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    acc_path = os.path.join(plots_dir, f'plot{log_number}_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: plot{log_number}_loss.png and plot{log_number}_accuracy.png")

def process_all_logs(logs_dir=LOGS_PATH, plots_dir=PLOTS_PATH):
    clear_plots_folder(plots_dir)
    
    if not os.path.exists(logs_dir):
        print(f"Error: '{logs_dir}/' directory not found!")
        return
    
    log_files = sorted([f for f in os.listdir(logs_dir) if f.startswith('log') and f.endswith('.txt')])
    
    if not log_files:
        print(f"No log files found in '{logs_dir}/' directory!")
        return
    
    print(f"\nFound {len(log_files)} log file(s)")
    
    for log_file in log_files:
        #regex log number from filename
        log_number = re.search(r'log(\d+)\.txt', log_file).group(1)
        
        filepath = os.path.join(logs_dir, log_file)
        print(f"\nProcessing {log_file}...")
        
        try:
            epochs, train_losses, train_accs, val_losses, val_accs, hyperparams = parse_log_file(filepath)
            create_plots(epochs, train_losses, train_accs, val_losses, val_accs, log_number, hyperparams, plots_dir)
            
        except Exception as e:
            print(f"  Error processing {log_file}: {str(e)}")
    
    print(f"All plots saved to '{plots_dir}/' directory")

if __name__ == "__main__":
    process_all_logs(logs_dir=LOGS_PATH, plots_dir=PLOTS_PATH)