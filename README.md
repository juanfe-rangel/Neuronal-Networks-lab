# Neural Networks Lab: Fashion MNIST Classification

## 1. Problem Description

**Objective:** Build and compare neural network architectures for multi-class image classification

**Task:** Classify 28×28 grayscale images of clothing items into 10 categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**Goal:** Understand why convolutional neural networks (CNNs) outperform fully-connected networks on image data through:
- Baseline model (Dense layers only)
- Custom CNN design with justified architectural choices
- Controlled experiments on kernel size and network depth
- Analysis of inductive biases

---

## 2. Dataset Description

**Fashion MNIST Dataset:**
- **60,000** training images + **10,000** test images
- **28×28 pixels**, grayscale (1 channel)
- **10 balanced classes** (6,000 samples each)
- Pixel values: 0-255

**Preprocessing:**
- Normalization: pixels scaled to [0, 1]
- Reshaping: `(28, 28, 1)` for CNNs, `(784,)` for baseline
- One-hot encoding for labels

---

## 3. Architecture Diagrams

### Baseline Model (Fully-Connected)

```
Input: 28×28 → Flatten → 784 features
         ↓
    Dense(256) + ReLU + Dropout(0.3)
         ↓
    Dense(128) + ReLU + Dropout(0.3)
         ↓
    Dense(10) + Softmax
```
- **Parameters:** ~235,000
- **Limitation:** Destroys spatial structure, parameter-inefficient

### Custom CNN Architecture

```
Input: 28×28×1
       ↓
[Block 1] Conv2D(32, 3×3, same) + ReLU → MaxPool(2×2) → 14×14×32
       ↓
[Block 2] Conv2D(64, 3×3, same) + ReLU → MaxPool(2×2) → 7×7×64
       ↓
Flatten → Dense(128) + ReLU + Dropout(0.5) → Dense(10) + Softmax
```
- **Parameters:** ~93,000 (60% fewer than baseline)
- **Advantages:** Preserves spatial structure, parameter sharing, hierarchical features

**Design Justifications:**
- **3×3 kernels:** Minimal size for directional patterns, parameter-efficient
- **2 conv layers:** Matches image complexity (28×28), learns edge→shape hierarchy
- **32→64 filters:** Progressive feature extraction
- **MaxPooling(2×2):** Controlled spatial reduction (28→14→7)

---

## Experimental Results

### Model Performance Comparison

| Model | Test Accuracy | Test Loss | Parameters | Training Time |
|-------|---------------|-----------|------------|---------------|
| **Baseline (Dense)** | 88-89% | 0.32-0.35 | ~235,000 | ~3 min |
| **CNN (3×3, 2 layers)** | 91-92% | 0.23-0.26 | ~93,000 | ~4 min |
| **CNN (5×5, 2 layers)** | 91-92% | 0.23-0.26 | ~120,000 | ~5 min |
| **CNN (1 layer)** | 88-89% | 0.30-0.33 | ~65,000 | ~2 min |
| **CNN (3 layers)** | 90-91% | 0.26-0.29 | ~115,000 | ~6 min |

### Key Findings

#### Experiment 1: Kernel Size (3×3 vs 5×5)
- **Result:** Similar accuracy (~91-92%)
- **Trade-off:** 3×3 kernels are more parameter-efficient
- **Conclusion:** 3×3 kernels optimal for small images

#### Experiment 2: Network Depth (1 vs 2 vs 3 layers)
- **1 layer:** Limited feature hierarchy, lower accuracy (~88%)
- **2 layers:** Optimal balance, best accuracy (~91-92%)
- **3 layers:** Diminishing returns, potential overfitting (~90-91%)
- **Conclusion:** 2 convolutional layers provide best balance for this dataset
4. Experimental Results

### Overall Performance

| Model | Test Accuracy | Parameters | Key Insight |
|-------|---------------|------------|-------------|
| **Baseline (Dense)** | 88-89% | ~235,000 | Reference point |
| **CNN (3×3, 2 layers)** | **91-92%** | ~93,000 | **Optimal** |
| **CNN (5×5, 2 layers)** | 91-92% | ~122,000 | Less efficient |
| **CNN (1 layer)** | 88-89% | ~86,000 | Too shallow |
| **CNN (3 layers)** | 90-91% | ~151,000 | Diminishing returns |

**CNN Improvement:** +3-4% accuracy with 60% fewer parameters
5. Interpretation

### Why CNNs Outperformed the Baseline (+3-4% accuracy, 60% fewer parameters)

**1. Spatial Structure Preservation**
- Baseline flattens 28×28 → destroys pixel relationships
- CNN maintains 2D structure → adjacent pixels stay connected

**2. Parameter Sharing (Translation Invariance)**
- Baseline: needs different weights for "shoe" at every position
- CNN: same 3×3 filter slides everywhere → learns once, applies everywhere

**3. Hierarchical Feature Learning**
- Baseline: all features at same level
- CNN: edges (layer 1) → shapes (layer 2) → objects (dense layers)

### Inductive Biases of Convolution

CNNs encode 3 assumptions about image data:

**1. Locality:** Nearby pixels are more related than distant pixels  
**2. Translation Equivariance:** A feature is useful everywhere in the image  
**3. Compositional Hierarchy:** Complex patterns built from simple ones

**Impact:** These biases reduce the hypothesis space → faster learning, better generalization

### When Convolution is NOT Appropriate

** Tabular Data** (age, income, zip) - no spatial structure  
** Graph Data** (social networks) - irregular connectivity  
** Long Sequences** (language) - limited receptive field  
** Permutation-Invariant Data** (sets) - order doesn't matter  
** Non-Uniform Importance** (position-specific meaning)

---

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib tensorflow seaborn scikit-learn pandas

# Open and run notebook
jupyter notebook neuronalNetworkLab.ipynb
```

**Notebook contains:**
1. Dataset Exploration (EDA)
2. Baseline Model (Dense layers)
3. CNN Architecture Design
4. Controlled Experiments
5. Interpretation & Analysis

---

## Authors
Juan Felipe Rangel Rodriguez
