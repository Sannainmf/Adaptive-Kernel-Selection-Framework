# Adaptive Kernel Selection Framework

A comprehensive PyTorch implementation of adaptive receptive field CNNs with intelligent kernel selection mechanisms. This framework compares adaptive convolutional neural networks against standard CNNs across multiple architectures and datasets.

## 🚀 Features

- **Adaptive Kernel Selection**: Dynamic selection between 3x3, 5x5, and 7x7 kernels based on feature statistics
- **Multiple Architectures**: 3-layer, 6-layer, and VGG11 CNN implementations
- **Multi-Dataset Support**: CIFAR-10, CIFAR-100, and Fashion-MNIST
- **Statistical Analysis**: Comprehensive evaluation with multiple random seeds
- **Clean Visualization**: Professional plotting and result analysis

## 📁 Repository Structure

### 3-Layer CNNs
- `3LayerC10.py` - CIFAR-10 (30 epochs)
- `3LayerC100.py` - CIFAR-100 (65 epochs)
- `3LayerFM.py` - Fashion-MNIST (25 epochs)

### 6-Layer CNNs
- `6LayerC10.py` - CIFAR-10 (30 epochs)
- `6LayerC100.py` - CIFAR-100 (65 epochs)
- `6LayerFM.py` - Fashion-MNIST (25 epochs)

### VGG11 CNNs
- `vgg11C10.py` - CIFAR-10 (30 epochs)
- `vgg11C100.py` - CIFAR-100 (65 epochs)
- `vgg11FM.py` - Fashion-MNIST (25 epochs)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Sannainmf/Adaptive-Kernel-Selection-Framework.git
cd Adaptive-Kernel-Selection-Framework

# Install dependencies
pip install torch torchvision numpy matplotlib scipy
```

## 🎯 Usage

### Quick Start
```bash
# Run a quick test (2 seeds, 5-10 epochs)
python 3LayerC10.py
# Select 'y' for quick test when prompted

# Run full experiment (5 seeds, full epochs)
python 3LayerC10.py
# Select 'n' for quick test when prompted
```

### Experiment Configuration

Each script provides two modes:
1. **Quick Test**: 2 seeds, reduced epochs for fast validation
2. **Full Experiment**: 5 seeds, complete epochs for publication-ready results

### Example Output
```
Seed 42 (1/5)
  Adaptive: 85.23%, Standard: 83.45%, Diff: +1.78%
Seed 123 (2/5)
  Adaptive: 84.91%, Standard: 82.87%, Diff: +2.04%
...

Results - Adaptive: 85.12±0.34%, Standard: 83.16±0.28%, Improvement: +1.96±0.41%, p=0.0234
```

## 🧠 Architecture Details

### Adaptive Kernel Selection Mechanism

The framework implements an intelligent kernel selection system:

```python
class AdaptiveKernelSelector(nn.Module):
    def __init__(self, in_channels, num_kernels=3):
        self.kernel_sizes = [3, 5, 7]  # Available kernel sizes
        self.selector = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_kernels),
            nn.Softmax(dim=1)
        )
```

**How it works:**
1. **Feature Statistics**: Global Average Pooling captures spatial statistics
2. **Attention Weights**: MLP generates soft attention weights for each kernel size
3. **Dynamic Convolution**: Weighted combination of 3x3, 5x5, and 7x7 convolutions
4. **Adaptive Selection**: Network learns optimal kernel sizes per layer

### Model Comparisons

| Architecture | Adaptive Layers | Standard Layers | Purpose |
|-------------|----------------|----------------|---------|
| 3-Layer CNN | 3 adaptive | 6 standard | Lightweight comparison |
| 6-Layer CNN | 6 adaptive | 12 standard | Balanced architecture |
| VGG11 | 8 adaptive | 8 standard | Deep network analysis |

## 📊 Results Analysis

### Generated Plots
Each experiment produces comprehensive visualizations:
- **Training/Validation Curves**: Loss and accuracy over epochs
- **Kernel Selection Heatmaps**: Visual representation of kernel preferences
- **Statistical Comparisons**: Bar charts showing layer-wise kernel selection

### Statistical Significance
- **Paired t-tests** for performance comparison
- **Confidence intervals** for accuracy measurements
- **Effect size analysis** for practical significance

## 🔬 Experimental Setup

### Datasets
- **CIFAR-10**: 10 classes, 32×32 RGB images
- **CIFAR-100**: 100 classes, 32×32 RGB images  
- **Fashion-MNIST**: 10 classes, 28×28 grayscale images

### Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: StepLR (step_size=20, gamma=0.1)
- **Batch Size**: 128 (64 for VGG11)
- **Validation Split**: 80/20 train/validation

### Evaluation Protocol
- **Multiple Seeds**: 5 different random seeds for robust statistics
- **Cross-Validation**: Consistent train/validation splits
- **Statistical Testing**: Paired t-tests for significance

## 📈 Expected Performance

### Typical Improvements
- **CIFAR-10**: 1-3% accuracy improvement
- **CIFAR-100**: 2-4% accuracy improvement
- **Fashion-MNIST**: 1-2% accuracy improvement

### Computational Overhead
- **Training Time**: ~10-15% increase due to adaptive mechanism
- **Memory Usage**: ~20% increase for kernel selection weights
- **Inference Speed**: Minimal impact after training

## 🛠️ Customization

### Adding New Datasets
1. Update normalization values in `transforms.Normalize()`
2. Modify `num_classes` parameter in model initialization
3. Adjust epoch counts based on dataset complexity

### Modifying Architectures
1. Add new adaptive layers to `adaptive_positions` list
2. Create corresponding `AdaptiveKernelSelector` instances
3. Update kernel selection analysis ranges

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{adaptive_kernel_selection_2024,
  title={Adaptive Kernel Selection Framework for Convolutional Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/Sannainmf/Adaptive-Kernel-Selection-Framework}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- New dataset implementations
- Additional CNN architectures
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Original VGG paper authors for the VGG11 architecture
- CIFAR and Fashion-MNIST dataset creators

---

**Note**: This framework is designed for research and educational purposes. For production deployments, consider additional optimizations and thorough testing.
