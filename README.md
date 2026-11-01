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
│   ├── chapter_2_neurons.py              # Single neurons & layers (manual + loop)
│   ├── chapter_2/neurons.py              # NumPy-powered neurons + batch processing
│   ├── chapter_3/layers.py               # Dense layer class + spiral data
│   ├── chapter_4/activations.py          # ReLU + Softmax
│   ├── chapter_5/loss.py                 # Categorical Cross-Entropy + accuracy
│   ├── chapter_6/optimizer.py            # Random search "optimization"
│   ├── chapter_7/derivatives.py          # Numerical derivatives
│   ├── chapter_9/back_prop.py            # Backpropagation, full training loop
│   ├── chapter_10/optimizers.py          # SGD, momentum, Adagrad, RMSprop, Adam
│   ├── chapter_11/testing_data.py        # Out-of-sample testing
│   ├── chapter_12/validation_data.py     # Validation data
│   ├── chapter_13/training_dataset.py    # Training dataset
│   ├── chapter_14/regularization.py      # L1 / L2 regularization
│   ├── chapter_15/dropout.py             # Dropout layer
│   ├── chapter_16/binary_logistic_regression.py  # Sigmoid + binary cross-entropy
│   ├── chapter_17/regression.py          # Linear activation, MSE / MAE
│   ├── chapter_18/model.py               # Model object
│   ├── chapter_19/real_dataset.py        # Fashion MNIST
│   ├── chapter_20/model_evaluation.py    # Model.evaluate()
│   ├── chapter_21/saving_loading.py      # Save / load params + whole model
│   └── chapter_22/predicting.py          # Model.predict() on new images
│
└── intro-notebooks/                      # Notebooks for chapters 1-9
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
| 7 | Derivatives | ✅ Done |
| 8 | Gradients, Partial Derivatives and the Chain Rule | ✅ Done |
| 9 | Backpropagation | ✅ Done |
| 10 | Optimizers | ✅ Done |
| 11 | Testing / Out-of-Sample Data | ✅ Done |
| 12 | Validation Data | ✅ Done |
| 13 | Training Dataset | ✅ Done |
| 14 | L1 and L2 Regularization | ✅ Done |
| 15 | Dropout | ✅ Done |
| 16 | Binary Logistic Regression | ✅ Done |
| 17 | Regression | ✅ Done |
| 18 | Model Object | ✅ Done |
| 19 | A Real Dataset (Fashion MNIST) | ✅ Done |
| 20 | Model Evaluation | ✅ Done |
| 21 | Saving and Loading Model Information | ✅ Done |
| 22 | Prediction / Inference | ✅ Done |

**Book finished!** 🎉

---------|-------|--------|
| 1 | Introduction to Neural Networks | ✅ Done |
| 2 | Coding Our First Neurons | ✅ Done |
| 3 | Adding Layers | ✅ Done |
| 4 | Activation Functions (ReLU, Softmax) | ✅ Done |
| 5 | Calculating Network Error with Loss | ✅ Done |
| 6 | Introducing Optimization | ✅ Done |
| 7 | Derivatives | ✅ Done |
| 8 | Gradients Partial Derivatives and the Chain Rule | ✅ Done |
| 7 | Backpropagation | 🔄 In Progress |
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

### Derivatives

- **Slope of the loss curve** — *“if I nudge this weight a tiny bit, how much does the loss change?”*
- **Gradient** = a vector of those slopes for every weight & bias
- **Chain rule** passes that slope backwards through the layers → that’s backpropagation
- **Weight update**: `weight -= learning_rate * gradient` (step opposite the slope to go downhill)
- Random search is dead. Slope * direction * tiny step. Rinse, repeat. ✨

### Backprop & Optimizers
- Backward pass for every layer, activation and loss
- Combined Softmax + Cross-Entropy backward (way faster gradient)
- **SGD** (with learning rate decay + momentum), **Adagrad**, **RMSprop**, **Adam**

### Generalization
- Out-of-sample testing, validation data, why you don't tune on test data
- **L1 / L2 regularization** on weights and biases
- **Dropout** (only active during training)

### Other kinds of problems
- **Binary logistic regression** — sigmoid + binary cross-entropy
- **Regression** — linear output, MSE / MAE loss, sine wave fitting

### The Model object
- `Model` class: `add()`, `set()`, `finalize()`, `train()` — Keras-style API built from scratch
- Mini-batch training, accumulated loss / accuracy per epoch
- `evaluate()` on any dataset, `predict()` for inference

### Real data — Fashion MNIST
- Loading 70k 28x28 images with OpenCV, scaling to [-1, 1], flattening, shuffling
- ~88% test accuracy with a 2 hidden layer (128 neurons) dense net
- Save / load just the parameters, or pickle the whole model
- Predicting on random images from the internet (resize + invert colors first!)


---

## 🛠️ Stack

- **Python 3.10**
- **NumPy** — the only ML dependency
- **nnfs** — for reproducible spiral dataset generation
- **Matplotlib** — for visualizations

---

## 🚀 Running the Code

```bash
pip install numpy nnfs matplotlib opencv-python
```

Then run any chapter script directly:

```bash
python 02_basic_neurons/chapter_5/loss.py
```

Chapters 19-22 use Fashion MNIST. Run them from the repo root — chapter 19 downloads + unzips the dataset,
chapter 21 saves `fashion_mnist.model`, and chapter 22 loads it to make predictions:

```bash
python 02_basic_neurons/chapter_21/saving_loading.py
python 02_basic_neurons/chapter_22/predicting.py
```

Or open the notebooks in Jupyter:

```bash
jupyter notebook intro-notebooks/
```

---

## 📈 Active Development

This repo is actively maintained and updated regularly — 133 contributions in the last year and counting. All 22 chapters are done — next up is building stuff on top of it.

---

*Following the [Neural Networks from Scratch](https://nnfs.io) book — but building understanding, not copying answers.*
