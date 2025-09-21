"""
VGG11 Adaptive Receptive Field CNN Experiment
Multi-seed experiment with comprehensive plotting
Modified for CIFAR-100 dataset with 65 epochs
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
    def __init__(self, num_classes=100):  # Changed to 100 for CIFAR-100
        super(AdaptiveVGG11, self).__init__()
       
        # VGG11 configuration: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        # Modified to include adaptive kernel selection
       
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.adaptive1 = AdaptiveKernelSelector(64)
        self.bn1_adaptive = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.adaptive2 = AdaptiveKernelSelector(128)
        self.bn2_adaptive = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.adaptive3 = AdaptiveKernelSelector(256)
        self.bn3_adaptive = nn.BatchNorm2d(256)
       
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive4 = AdaptiveKernelSelector(256)
        self.bn4_adaptive = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.adaptive5 = AdaptiveKernelSelector(512)
        self.bn5_adaptive = nn.BatchNorm2d(512)
       
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.adaptive6 = AdaptiveKernelSelector(512)
        self.bn6_adaptive = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 5
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.adaptive7 = AdaptiveKernelSelector(512)
        self.bn7_adaptive = nn.BatchNorm2d(512)
       
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.adaptive8 = AdaptiveKernelSelector(512)
        self.bn8_adaptive = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),  # Changed to num_classes for CIFAR-100
        )
       
        self.attention_weights = []
       
    def forward(self, x):
        self.attention_weights = []
       
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x, weights1 = self.adaptive1(x)
        x = F.relu(self.bn1_adaptive(x))
        x = self.pool1(x)
        self.attention_weights.append(weights1)
       
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x, weights2 = self.adaptive2(x)
        x = F.relu(self.bn2_adaptive(x))
        x = self.pool2(x)
        self.attention_weights.append(weights2)
       
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x, weights3 = self.adaptive3(x)
        x = F.relu(self.bn3_adaptive(x))
        self.attention_weights.append(weights3)
       
        x = F.relu(self.bn4(self.conv4(x)))
        x, weights4 = self.adaptive4(x)
        x = F.relu(self.bn4_adaptive(x))
        x = self.pool3(x)
        self.attention_weights.append(weights4)
       
        # Block 4
        x = F.relu(self.bn5(self.conv5(x)))
        x, weights5 = self.adaptive5(x)
        x = F.relu(self.bn5_adaptive(x))
        self.attention_weights.append(weights5)
       
        x = F.relu(self.bn6(self.conv6(x)))
        x, weights6 = self.adaptive6(x)
        x = F.relu(self.bn6_adaptive(x))
        x = self.pool4(x)
        self.attention_weights.append(weights6)
       
        # Block 5
        x = F.relu(self.bn7(self.conv7(x)))
        x, weights7 = self.adaptive7(x)
        x = F.relu(self.bn7_adaptive(x))
        self.attention_weights.append(weights7)
       
        x = F.relu(self.bn8(self.conv8(x)))
        x, weights8 = self.adaptive8(x)
        x = F.relu(self.bn8_adaptive(x))
        x = self.pool5(x)
        self.attention_weights.append(weights8)
       
        # Classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)
       
        return x

class StandardVGG11(nn.Module):
    def __init__(self, num_classes=100):  # Changed to 100 for CIFAR-100
        super(StandardVGG11, self).__init__()
       
        # Standard VGG11 configuration: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
       
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Block 5
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),  # Changed to num_classes for CIFAR-100
        )
       
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
       
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
       
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
       
        # Block 4
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool4(x)
       
        # Block 5
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool5(x)
       
        # Classifier
        x = torch.flatten(x, 1)
        x = self.classifier(x)
       
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model_single_run(model, train_loader, val_loader, num_epochs=65, device='cuda', verbose=False):  # Changed default to 65
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Adjusted for longer training
   
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
           
            if verbose and batch_idx % 100 == 0 and epoch % 10 == 0:  # Adjusted for longer training
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
       
        if verbose and epoch % 10 == 0:  # Adjusted for longer training
            print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
       
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
    # VGG11 has 8 adaptive layers
    kernel_selections = {i: [] for i in range(8)}
   
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
           
            for layer_idx, weights in enumerate(model.attention_weights):
                if layer_idx < 8:
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
    plt.plot(epochs, adaptive_train_loss_avg, color='#1f77b4', linewidth=2, label='Adaptive VGG11')
    plt.plot(epochs, standard_train_loss_avg, color='#ff7f0e', linewidth=2, label='Standard VGG11')
   
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (CIFAR-100)')
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
    plt.title('Validation Accuracy (CIFAR-100)')
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
    plt.title('Training Accuracy (CIFAR-100)')
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
    plt.title('Validation Loss (CIFAR-100)')
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
   
    im = plt.imshow(avg_selections, cmap='Blues', aspect='auto')
    plt.xticks(range(3), ['3x3', '5x5', '7x7'])
    plt.yticks(range(8), [f'Layer {i+1}' for i in range(8)])
    plt.title('VGG11 Kernel Selection Patterns (CIFAR-100)')
   
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
    plt.title('VGG11 Kernel Selection Patterns (First 5 Layers, CIFAR-100)')
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
    print(f"\nSUMMARY STATISTICS (CIFAR-100):")
    print(f"Adaptive VGG11:  {adaptive_mean:.2f}% ± {adaptive_std:.2f}%")
    print(f"Standard VGG11:  {standard_mean:.2f}% ± {standard_std:.2f}%")
    print(f"Improvement:     {improvement_mean:.2f}% ± {improvement_std:.2f}%")
    print(f"Statistical significance: t={t_stat:.3f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("NOT statistically significant (p >= 0.05)")

def run_experiment(seeds=[42, 123, 456, 789, 999], num_epochs=65):  # Changed default to 65
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
   
    # Changed to CIFAR-100
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
   
    results = {
        'adaptive_accuracies': [],
        'standard_accuracies': [],
        'improvements': [],
        'adaptive_kernel_selections': [],
        'adaptive_training_curves': [],
        'standard_training_curves': [],
        'seeds': seeds
    }
   
    for i, seed in enumerate(seeds):
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        
        set_seed(seed)
       
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])
       
        train_loader = DataLoader(train_dataset_split, batch_size=64, shuffle=True)  # Smaller batch size for VGG11
        val_loader = DataLoader(val_dataset_split, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
       
        adaptive_model = AdaptiveVGG11(num_classes=100)
        adaptive_metrics = train_model_single_run(adaptive_model, train_loader, val_loader, num_epochs, device, verbose=(i==0))
        adaptive_test_acc = test_model_single_run(adaptive_model, test_loader, device)
        adaptive_kernel_sel = analyze_kernel_selection_single_run(adaptive_model, test_loader, device)
        
        standard_model = StandardVGG11(num_classes=100)
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
        with open('results/vgg11_cifar100_results.json', 'w') as f:
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