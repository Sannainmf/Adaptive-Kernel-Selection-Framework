"""
VGG11 Adaptive Receptive Field CNN Experiment
Multi-seed experiment with comprehensive plotting
Modified for CIFAR-10 dataset with 30 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

class AdaptiveKernelSelector(nn.Module):
    def __init__(self, in_channels, num_kernels=3):
        super(AdaptiveKernelSelector, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_sizes = [3, 5, 7]
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_kernels),
            nn.Softmax(dim=1)
        )
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2)
            for k in self.kernel_sizes
        ])
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        stats = self.gap(x).view(batch_size, channels)
        weights = self.selector(stats)
        
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        output = torch.zeros_like(x)
        for i, conv_out in enumerate(conv_outputs):
            weight = weights[:, i].view(batch_size, 1, 1, 1)
            output += weight * conv_out
            
        return output, weights

class AdaptiveVGG11(nn.Module):
    def __init__(self, num_classes=10):  # Changed to 10 for CIFAR-10
        super(AdaptiveVGG11, self).__init__()
        
        # VGG11 architecture with adaptive kernel selection
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.adaptive_layers = nn.ModuleList([
            AdaptiveKernelSelector(64),
            AdaptiveKernelSelector(128),
            AdaptiveKernelSelector(256),
            AdaptiveKernelSelector(512),
            AdaptiveKernelSelector(512),
            AdaptiveKernelSelector(512),
            AdaptiveKernelSelector(512),
            AdaptiveKernelSelector(512),
        ])
        
        self.adaptive_positions = [0, 1, 2, 3, 4, 5, 6, 7]  # After each conv layer
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),  # Changed to num_classes for CIFAR-10
        )
        
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        adaptive_idx = 0
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Apply adaptive kernel selection after conv layers
            if i in self.adaptive_positions and adaptive_idx < len(self.adaptive_layers):
                x, weights = self.adaptive_layers[adaptive_idx](x)
                self.attention_weights.append(weights)
                adaptive_idx += 1
        
        x = self.classifier(x)
        return x

class StandardVGG11(nn.Module):
    def __init__(self, num_classes=10):  # Changed to 10 for CIFAR-10
        super(StandardVGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),  # Changed to num_classes for CIFAR-10
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model_single_run(model, train_loader, val_loader, num_epochs=30, device='cuda', verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if verbose and batch_idx % 200 == 0 and epoch % 5 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss = criterion(outputs, target)
                val_running_loss += val_loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if verbose and epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%, Loss: {train_loss:.4f}')
        
        scheduler.step()
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def test_model_single_run(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def analyze_kernel_selection_single_run(model, test_loader, device='cuda'):
    model.eval()
    kernel_selections = {i: [] for i in range(8)}  # 8 adaptive layers for VGG11
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
            
            for layer_idx, weights in enumerate(model.attention_weights):
                if layer_idx < 8:  # 8 adaptive layers
                    avg_weights = weights.mean(dim=0)
                    kernel_selections[layer_idx].append(avg_weights.cpu().numpy())
    
    for layer_idx in kernel_selections:
        if kernel_selections[layer_idx]:
            kernel_selections[layer_idx] = np.mean(kernel_selections[layer_idx], axis=0)
        else:
            kernel_selections[layer_idx] = np.zeros(3)
    
    return kernel_selections

def plot_results(results):
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        from scipy import stats
    
    # Calculate statistics for summary
    adaptive_mean = np.mean(results['adaptive_accuracies'])
    adaptive_std = np.std(results['adaptive_accuracies'])
    standard_mean = np.mean(results['standard_accuracies'])
    standard_std = np.std(results['standard_accuracies'])
    improvements = results['improvements']
    improvement_mean = np.mean(improvements)
    improvement_std = np.std(improvements)
    t_stat, p_value = stats.ttest_rel(results['adaptive_accuracies'], results['standard_accuracies'])
    
    # Create separate figures for each plot
    os.makedirs('results', exist_ok=True)
    
    # Get epochs for training curves
    epochs = range(1, len(results['adaptive_training_curves'][0]['train_losses']) + 1)
    
    # Calculate averaged curves for each metric
    adaptive_train_losses = [curve['train_losses'] for curve in results['adaptive_training_curves']]
    standard_train_losses = [curve['train_losses'] for curve in results['standard_training_curves']]
    adaptive_train_accs = [curve['train_accuracies'] for curve in results['adaptive_training_curves']]
    standard_train_accs = [curve['train_accuracies'] for curve in results['standard_training_curves']]
    adaptive_val_losses = [curve['val_losses'] for curve in results['adaptive_training_curves']]
    standard_val_losses = [curve['val_losses'] for curve in results['standard_training_curves']]
    adaptive_val_accs = [curve['val_accuracies'] for curve in results['adaptive_training_curves']]
    standard_val_accs = [curve['val_accuracies'] for curve in results['standard_training_curves']]
    
    # Average across seeds
    adaptive_train_loss_avg = np.mean(adaptive_train_losses, axis=0)
    standard_train_loss_avg = np.mean(standard_train_losses, axis=0)
    adaptive_train_acc_avg = np.mean(adaptive_train_accs, axis=0)
    standard_train_acc_avg = np.mean(standard_train_accs, axis=0)
    adaptive_val_loss_avg = np.mean(adaptive_val_losses, axis=0)
    standard_val_loss_avg = np.mean(standard_val_losses, axis=0)
    adaptive_val_acc_avg = np.mean(adaptive_val_accs, axis=0)
    standard_val_acc_avg = np.mean(standard_val_accs, axis=0)
    
    # Plot 1: Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_train_loss_avg, color='#1f77b4', linewidth=2, label='Adaptive VGG11')
    plt.plot(epochs, standard_train_loss_avg, color='#ff7f0e', linewidth=2, label='Standard VGG11')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_val_acc_avg, color='#1f77b4', linewidth=2, label='Adaptive VGG11')
    plt.plot(epochs, standard_val_acc_avg, color='#ff7f0e', linewidth=2, label='Standard VGG11')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/validation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Training Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_train_acc_avg, color='#1f77b4', linewidth=2, label='Adaptive VGG11')
    plt.plot(epochs, standard_train_acc_avg, color='#ff7f0e', linewidth=2, label='Standard VGG11')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_val_loss_avg, color='#1f77b4', linewidth=2, label='Adaptive VGG11')
    plt.plot(epochs, standard_val_loss_avg, color='#ff7f0e', linewidth=2, label='Standard VGG11')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss (CIFAR-10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/validation_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: Kernel Selection Heatmap (8 layers for VGG11)
    plt.figure(figsize=(12, 8))
    avg_selections = np.zeros((8, 3))
    for kernel_sel in results['adaptive_kernel_selections']:
        for layer_idx in range(8):
            avg_selections[layer_idx] = kernel_sel[layer_idx]
    avg_selections /= len(results['adaptive_kernel_selections'])
    
    im = plt.imshow(avg_selections, cmap='viridis', aspect='auto')
    plt.xticks(range(3), ['3x3', '5x5', '7x7'])
    plt.yticks(range(8), [f'Layer {i+1}' for i in range(8)])
    plt.title('VGG11 Kernel Selection Patterns (CIFAR-10)')
    
    for i in range(8):
        for j in range(3):
            plt.text(j, i, f'{avg_selections[i, j]:.3f}', ha="center", va="center", 
                    color="black", fontweight='bold')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('results/kernel_selection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 6: Kernel Selection Bar Chart (showing first 5 layers to avoid cluttering)
    plt.figure(figsize=(12, 6))
    
    # Calculate average kernel selections for bar chart (first 5 layers)
    layer_names = [f'Layer {i+1}' for i in range(5)]
    kernel_3x3 = [avg_selections[i][0] for i in range(5)]
    kernel_5x5 = [avg_selections[i][1] for i in range(5)]
    kernel_7x7 = [avg_selections[i][2] for i in range(5)]
    
    x = np.arange(len(layer_names))
    width = 0.25
    
    plt.bar(x - width, kernel_3x3, width, label='3x3', color='#1f77b4', alpha=0.8)
    plt.bar(x, kernel_5x5, width, label='5x5', color='#ff7f0e', alpha=0.8)
    plt.bar(x + width, kernel_7x7, width, label='7x7', color='#2ca02c', alpha=0.8)
    
    plt.xlabel('Layer')
    plt.ylabel('Selection Weight')
    plt.title('VGG11 Kernel Selection Patterns (First 5 Layers, CIFAR-10)')
    plt.xticks(x, layer_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/kernel_selection_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"All plots saved as separate PNG files in 'results/' directory:")
    print(f"  - training_loss.png")
    print(f"  - validation_accuracy.png")
    print(f"  - training_accuracy.png")
    print(f"  - validation_loss.png")
    print(f"  - kernel_selection_heatmap.png")
    print(f"  - kernel_selection_bars.png")
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS (CIFAR-10):")
    print(f"Adaptive VGG11:  {adaptive_mean:.2f}% ± {adaptive_std:.2f}%")
    print(f"Standard VGG11:  {standard_mean:.2f}% ± {standard_std:.2f}%")
    print(f"Improvement:     {improvement_mean:.2f}% ± {improvement_std:.2f}%")
    print(f"Statistical significance: t={t_stat:.3f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("NOT statistically significant (p >= 0.05)")

def run_experiment(seeds=[42, 123, 456, 789, 999], num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    
    # Changed to CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    results = {
        'adaptive_accuracies': [],
        'adaptive_kernel_selections': [],
        'adaptive_training_curves': [],
        'standard_accuracies': [],
        'standard_training_curves': [],
        'improvements': []
    }
    
    for i, seed in enumerate(seeds):
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        
        set_seed(seed)
        
        # Create train/validation split
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset_split, batch_size=64, shuffle=True)  # Smaller batch size for VGG11
        val_loader = DataLoader(val_dataset_split, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        adaptive_model = AdaptiveVGG11(num_classes=10)
        adaptive_metrics = train_model_single_run(adaptive_model, train_loader, val_loader, num_epochs, device, verbose=(i==0))
        adaptive_test_acc = test_model_single_run(adaptive_model, test_loader, device)
        adaptive_kernel_sel = analyze_kernel_selection_single_run(adaptive_model, test_loader, device)
        
        standard_model = StandardVGG11(num_classes=10)
        standard_metrics = train_model_single_run(standard_model, train_loader, val_loader, num_epochs, device, verbose=False)
        standard_test_acc = test_model_single_run(standard_model, test_loader, device)
        
        improvement = adaptive_test_acc - standard_test_acc
        results['adaptive_accuracies'].append(adaptive_test_acc)
        results['standard_accuracies'].append(standard_test_acc)
        results['improvements'].append(improvement)
        results['adaptive_kernel_selections'].append(adaptive_kernel_sel)
        results['adaptive_training_curves'].append(adaptive_metrics)
        results['standard_training_curves'].append(standard_metrics)
        
        print(f"  Adaptive: {adaptive_test_acc:.2f}%, Standard: {standard_test_acc:.2f}%, Diff: {improvement:+.2f}%")
    
    # Calculate final statistics
    adaptive_mean = np.mean(results['adaptive_accuracies'])
    adaptive_std = np.std(results['adaptive_accuracies'])
    standard_mean = np.mean(results['standard_accuracies'])
    standard_std = np.std(results['standard_accuracies'])
    improvement_mean = np.mean(results['improvements'])
    improvement_std = np.std(results['improvements'])
    
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(results['adaptive_accuracies'], results['standard_accuracies'])
        print(f"\nResults - Adaptive: {adaptive_mean:.2f}±{adaptive_std:.2f}%, Standard: {standard_mean:.2f}±{standard_std:.2f}%, Improvement: {improvement_mean:.2f}±{improvement_std:.2f}%, p={p_value:.4f}")
    except ImportError:
        print("Scipy not available for statistical tests")
    
    plot_results(results)
    
    try:
        os.makedirs('results', exist_ok=True)
        with open('results/vgg11_cifar10_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Could not save results: {e}")
    
    return results

if __name__ == "__main__":
    test_mode = input("Quick test first? (y/n): ").lower().strip()
    
    if test_mode == 'y':
        results = run_experiment(seeds=[42, 123], num_epochs=5)
        if input("Run full experiment? (y/n): ").lower().strip() != 'y':
            exit()
    
    results = run_experiment(seeds=[42, 123, 456, 789, 999], num_epochs=30)
