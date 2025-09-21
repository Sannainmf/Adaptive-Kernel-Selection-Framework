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

class AdaptiveReceptiveFieldCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(AdaptiveReceptiveFieldCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.adaptive1 = AdaptiveKernelSelector(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive2 = AdaptiveKernelSelector(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.adaptive3 = AdaptiveKernelSelector(256)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x, weights1 = self.adaptive1(x)
        x = F.relu(self.bn2(x))
        self.attention_weights.append(weights1)
        
        # Block 2
        x = F.relu(self.bn3(self.conv2(x)))
        x, weights2 = self.adaptive2(x)
        x = F.relu(self.bn4(x))
        self.attention_weights.append(weights2)
        
        # Block 3
        x = F.relu(self.bn5(self.conv3(x)))
        x, weights3 = self.adaptive3(x)
        x = F.relu(self.bn6(x))
        self.attention_weights.append(weights3)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class StandardCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(StandardCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv1b(x)))
        
        # Block 2
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv2b(x)))
        
        # Block 3
        x = F.relu(self.bn5(self.conv3(x)))
        x = F.relu(self.bn6(self.conv3b(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model_single_run(model, train_loader, val_loader, num_epochs=65, device='cuda', verbose=False):
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
    kernel_selections = {0: [], 1: [], 2: []}  # 3 adaptive layers
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
            
            for layer_idx, weights in enumerate(model.attention_weights):
                if layer_idx < 3:  # 3 adaptive layers
                    avg_weights = weights.mean(dim=0)
                    kernel_selections[layer_idx].append(avg_weights.cpu().numpy())
    
    for layer_idx in kernel_selections:
        kernel_selections[layer_idx] = np.mean(kernel_selections[layer_idx], axis=0)
    
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
    plt.plot(epochs, adaptive_train_loss_avg, color='#1f77b4', linewidth=2, label='Adaptive CNN')
    plt.plot(epochs, standard_train_loss_avg, color='#ff7f0e', linewidth=2, label='Standard CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_val_acc_avg, color='#1f77b4', linewidth=2, label='Adaptive CNN')
    plt.plot(epochs, standard_val_acc_avg, color='#ff7f0e', linewidth=2, label='Standard CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/validation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Training Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_train_acc_avg, color='#1f77b4', linewidth=2, label='Adaptive CNN')
    plt.plot(epochs, standard_train_acc_avg, color='#ff7f0e', linewidth=2, label='Standard CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, adaptive_val_loss_avg, color='#1f77b4', linewidth=2, label='Adaptive CNN')
    plt.plot(epochs, standard_val_loss_avg, color='#ff7f0e', linewidth=2, label='Standard CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/validation_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: Kernel Selection Analysis
    plt.figure(figsize=(12, 4))
    kernel_names = ['3x3', '5x5', '7x7']
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3']
    
    avg_kernel_selections = np.mean(results['adaptive_kernel_selections'], axis=0)
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.bar(kernel_names, avg_kernel_selections[i], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title(f'{layer_names[i]} Kernel Selection')
        plt.ylabel('Average Weight')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/kernel_selection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\nResults - Adaptive: {adaptive_mean:.2f}±{adaptive_std:.2f}%, Standard: {standard_mean:.2f}±{standard_std:.2f}%, Improvement: {improvement_mean:.2f}±{improvement_std:.2f}%, p={p_value:.4f}")

def run_experiment(seeds=[42, 123, 456, 789, 999], num_epochs=65):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
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
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        adaptive_model = AdaptiveReceptiveFieldCNN(num_classes=100)
        adaptive_metrics = train_model_single_run(adaptive_model, train_loader, val_loader, num_epochs, device, verbose=(i==0))
        adaptive_test_acc = test_model_single_run(adaptive_model, test_loader, device)
        adaptive_kernel_sel = analyze_kernel_selection_single_run(adaptive_model, test_loader, device)
        
        results['adaptive_accuracies'].append(adaptive_test_acc)
        results['adaptive_kernel_selections'].append(adaptive_kernel_sel)
        results['adaptive_training_curves'].append(adaptive_metrics)
        
        standard_model = StandardCNN(num_classes=100)
        standard_metrics = train_model_single_run(standard_model, train_loader, val_loader, num_epochs, device, verbose=False)
        standard_test_acc = test_model_single_run(standard_model, test_loader, device)
        
        results['standard_accuracies'].append(standard_test_acc)
        results['standard_training_curves'].append(standard_metrics)
        
        improvement = adaptive_test_acc - standard_test_acc
        results['improvements'].append(improvement)
        
        print(f"  Adaptive: {adaptive_test_acc:.2f}%, Standard: {standard_test_acc:.2f}%, Diff: {improvement:+.2f}%")
    
    plot_results(results)
    
    try:
        os.makedirs('results', exist_ok=True)
        with open('results/cifar100_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Could not save results: {e}")
    
    return results

if __name__ == "__main__":
    test_mode = input("Quick test first? (y/n): ").lower().strip()
    
    if test_mode == 'y':
        results = run_experiment(seeds=[42, 123], num_epochs=10)
        if input("Run full experiment? (y/n): ").lower().strip() != 'y':
            exit()
    
    results = run_experiment(seeds=[42, 123, 456, 789, 999], num_epochs=65)
