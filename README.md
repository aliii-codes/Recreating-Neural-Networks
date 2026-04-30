# 🧠 Recreating Neural Networks from Scratch

> *"Just a normal teen with not so normal interests"*

A hands-on journey to build a complete neural network framework from scratch using only **NumPy** — no PyTorch, no TensorFlow, no shortcuts.

---

## 🎯 Goal

Most people learn ML by calling `.fit()`. This repo is about understanding what actually happens *inside* — building every component from first principles so that when a model trains, updates weights, and minimizes loss, nothing is magic anymore.

By the end of this project, the goal is to have built:
- A working neural network library from scratch
- Deep understanding of backpropagation and gradients
- The ability to train models without relying on external ML frameworks

---

## 📁 Project Structure

```
.
├── 02_basic_neurons/
│   ├── chapter_2_neurons.py        # Single neurons & layers (manual + loop)
│   ├── chapter_2/
│   │   └── neurons.py              # NumPy-powered neurons + batch processing
│   ├── chapter_3/
│   │   └── layers.py               # Dense layer class + spiral data
│   ├── chapter_4/
│   │   └── activations.py          # ReLU + Softmax implementations
│   └── chapter_5/
│       └── loss.py                 # Categorical Cross-Entropy loss + accuracy
│
└── intro-notebooks/
    ├── 01_intro-to-neural-networks.ipynb
    ├── 02_coding-our-first-neurons.ipynb
    ├── 03_adding-layers.ipynb
    ├── 04-activation-functions.ipynb
    ├── 05-calculating-network-error-with-loss.ipynb
    └── 06-introducing-optimization.ipynb      ← Done 
```

---

## 📖 Progress

| Chapter | Topic | Status |
|---------|-------|--------|
| 1 | Introduction to Neural Networks | ✅ Done |
| 2 | Coding Our First Neurons | ✅ Done |
| 3 | Adding Layers | ✅ Done |
| 4 | Activation Functions (ReLU, Softmax) | ✅ Done |
| 5 | Calculating Network Error with Loss | ✅ Done |
| 6 | Introducing Optimization | ✅ Done |
| 7 | Derivatives | 🔄 In Progress |

---

## 🧩 What's Been Built

### Neurons & Layers
- Single neuron: `output = sum(inputs × weights) + bias`
- Layer of neurons with manual calculation, loops, and NumPy
- Batch processing with matrix operations (`np.dot` + transpose)
- `Layer_Dense` class with proper weight/bias initialization

### Activation Functions
- **ReLU** — `max(0, x)`, kills negatives, introduces non-linearity
- **Softmax** — converts raw outputs into a probability distribution, with overflow protection via max-subtraction

### Loss Functions
- **Categorical Cross-Entropy** — `-log(predicted_probability_of_correct_class)`
- Handles both sparse labels and one-hot encoded targets
- Clipping (`1e-7`) to prevent `log(0)` blowing up to infinity
- Accuracy calculation via `argmax` comparison

### Optimization
- Random search baseline — brute force weight tweaking
- Concept of loss-guided adjustment
- Derivatives chapter in progress — foundation for gradient descent

---

## 🛠️ Stack

- **Python 3.10**
- **NumPy** — the only ML dependency
- **nnfs** — for reproducible spiral dataset generation
- **Matplotlib** — for visualizations

---

## 🚀 Running the Code

```bash
pip install numpy nnfs matplotlib
```

Then run any chapter script directly:

```bash
python 02_basic_neurons/chapter_5/loss.py
```

Or open the notebooks in Jupyter:

```bash
jupyter notebook intro-notebooks/
```

---

## 📈 Active Development

This repo is actively maintained and updated regularly — 133 contributions in the last year and counting. New chapters drop as they're completed.

---

*Following the [Neural Networks from Scratch](https://nnfs.io) book — but building understanding, not copying answers.*
