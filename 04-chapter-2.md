# Chapter 2: The Atom of Intelligence: Perceptrons and Forces

## In This Chapter
- Comparing Biological and Digital Neurons
- The Physics of Weights, Biases, and Activation Functions
- Activation Functions as Phase Transitions
- Building Logic Gates from Neurons
- The XOR Problem and the Limits of Single Neurons
- Understanding Decision Boundaries
- Real-world Pattern Classification

---

## The Building Block of Intelligence

In Chapter 1, we learned that intelligence is energy minimization. But what exactly is being minimized? What structure performs the computation?

If Machine Learning is the study of materials, the **Neuron** is the atom. Everything in modern AI—from Siri to GPT-4 to AlphaFold—is made of these tiny processing units chained together in vast networks.

A single neuron is beautifully simple. It takes inputs, multiplies them by weights, adds a bias, and passes the result through an activation function. Yet from this simplicity emerges all of machine intelligence.

**THEORY LAB:** The perceptron is the digital equivalent of a biological neuron. But while biological neurons are analog devices running on electrochemical gradients, artificial neurons are pure mathematical functions. They process information, not electricity.

---

## Biological vs. Digital Neurons

### The Biological Neuron

A biological neuron has three main components:

1. **Dendrites:** Input receivers (like antenna)
2. **Cell Body (Soma):** Integration center (processes signals)
3. **Axon:** Output transmitter (sends signal to other neurons)

When enough electrical potential accumulates in the soma (exceeding a threshold), the neuron "fires"—sending an action potential down the axon.

### The Artificial Neuron (Perceptron)

The digital version captures this behavior mathematically:

**y = f(Σwᵢ·xᵢ + b)**

Or in vector notation:

**y = f(w^T·x + b)**

Let's translate each component into physics terms:

- **Inputs** (x₁, x₂, ..., xₙ): The incoming signals or forces acting on the system
- **Weights** (w₁, w₂, ..., wₙ): Coupling constants that determine signal strength
  - If wᵢ is large → strong connection (high conductivity)
  - If wᵢ ≈ 0 → weak connection (insulator)
  - If wᵢ < 0 → inhibitory connection (negative feedback)
- **Bias** (b): The threshold energy or activation offset
- **Activation Function** (f): The phase transition or nonlinearity

**REMEMBER:** The weighted sum Σwᵢxᵢ + b is a linear operation—it defines a hyperplane in input space. The activation function introduces nonlinearity, allowing neurons to model complex relationships.

---

## The Physics of Weights and Biases

### Weights as Force Multipliers

Imagine three forces acting on an object:
- Force 1: 10 N pointing East
- Force 2: 5 N pointing North  
- Force 3: 3 N pointing West

The net force is: **F_net = 10î + 5ĵ - 3î = 7î + 5ĵ**

A neuron performs the same calculation:

**z = w₁x₁ + w₂x₂ + w₃x₃ + b**

Where:
- xᵢ are the force magnitudes
- wᵢ are the directional components
- b is a constant bias force

### Bias: The Threshold Shift

The bias term allows the neuron to fire even when all inputs are zero. 

**Physical Analogy:** Think of a spring-loaded trigger. The bias determines how much force is needed to pull the trigger. A high positive bias means the trigger is "hair-trigger sensitive"—it fires easily. A large negative bias means the trigger is stiff—you need strong input signals to activate it.

### Geometric Interpretation

For a 2D input (x₁, x₂), the equation:

**w₁x₁ + w₂x₂ + b = 0**

defines a **line** in 2D space. This line is the **decision boundary**. Points on one side get classified as "1", points on the other side as "0".

For 3D inputs, it's a plane. For n-dimensional inputs, it's a hyperplane.

---

## Activation Functions: Phase Transitions

A purely linear neuron is useless for complex problems. To understand why, consider this:

If f(z) = z (identity function), then stacking multiple layers gives:

**y = W₃(W₂(W₁x)) = (W₃W₂W₁)x = W_combined·x**

No matter how many layers you stack, you just get one big linear transformation. You haven't gained any expressiveness.

**We need nonlinearity.**

### The Step Function (Heaviside Function)

The original perceptron used:

```
f(z) = { 1  if z ≥ 0
       { 0  if z < 0
```

**Physics Analogy:** A light switch. Below threshold → off. Above threshold → on.

**Problem:** Not differentiable at z = 0. Gradient descent breaks here because there's no gradient to follow.

### Sigmoid Function

The modern solution:

**σ(z) = 1/(1 + e^(-z))**

Properties:
- Smooth and differentiable everywhere
- Output range: (0, 1) → Interpretable as probability
- Derivative: σ'(z) = σ(z)(1 - σ(z)) (convenient for backpropagation)

**Physics Analogy:** A phase transition like ice melting to water. At z = 0, you're at the transition point. Far from zero, you're fully solid (0) or fully liquid (1).

**Limitations:** 
- **Vanishing Gradients:** For large |z|, the gradient σ'(z) ≈ 0. This causes learning to stall in deep networks.
- **Not Zero-Centered:** Outputs are always positive, which can slow convergence.

### Hyperbolic Tangent (tanh)

**tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))**

Properties:
- Output range: (-1, 1) → Zero-centered
- Steeper gradient than sigmoid near z = 0
- Still suffers from vanishing gradients for large |z|

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z) = { z  if z > 0
                      { 0  if z ≤ 0
```

**Physics Analogy:** A diode in an electrical circuit. Current flows in one direction (forward bias) but is blocked in the reverse direction.

**Why It Dominates Modern Deep Learning:**
1. **No Vanishing Gradient Problem:** For z > 0, gradient is constant (1)
2. **Computational Efficiency:** Just a threshold operation
3. **Sparsity:** Many neurons output exactly 0, creating sparse representations

**Limitation:** 
- **Dying ReLU Problem:** If a neuron's weights push z permanently negative, gradient becomes 0 forever and the neuron never updates. Solution: Use variants like Leaky ReLU.

---

## Lab 2.1: Visualizing Activation Functions

Let's see these phase transitions in action.

```python
"""
Lab 2.1: Activation Function Comparison
Visualizing different nonlinearities and their derivatives
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Define input range
z = torch.linspace(-6, 6, 1000)

# Define activation functions
sigmoid = torch.sigmoid(z)
tanh = torch.tanh(z)
relu = torch.relu(z)
leaky_relu = nn.functional.leaky_relu(z, negative_slope=0.1)

# Calculate derivatives manually for visualization
def sigmoid_derivative(z):
    s = torch.sigmoid(z)
    return s * (1 - s)

def tanh_derivative(z):
    return 1 - torch.tanh(z)**2

def relu_derivative(z):
    return (z > 0).float()

def leaky_relu_derivative(z):
    return torch.where(z > 0, torch.ones_like(z), torch.ones_like(z) * 0.1)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Activation functions
activations = [
    (sigmoid, 'Sigmoid σ(z)', 'blue'),
    (tanh, 'Tanh', 'green'),
    (relu, 'ReLU', 'red'),
    (leaky_relu, 'Leaky ReLU', 'purple')
]

for idx, (func, name, color) in enumerate(activations):
    axes[0, idx].plot(z.numpy(), func.numpy(), color=color, linewidth=2.5)
    axes[0, idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0, idx].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    axes[0, idx].set_xlabel('z (Input)', fontsize=10)
    axes[0, idx].set_ylabel('f(z) (Output)', fontsize=10)
    axes[0, idx].set_title(f'{name}', fontsize=11, fontweight='bold')
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].set_ylim(-1.5, 1.5)

# Row 2: Derivatives
derivatives = [
    (sigmoid_derivative(z), "σ'(z)", 'blue'),
    (tanh_derivative(z), "tanh'(z)", 'green'),
    (relu_derivative(z), "ReLU'(z)", 'red'),
    (leaky_relu_derivative(z), "LeakyReLU'(z)", 'purple')
]

for idx, (deriv, name, color) in enumerate(derivatives):
    axes[1, idx].plot(z.numpy(), deriv.numpy(), color=color, linewidth=2.5)
    axes[1, idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, idx].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    axes[1, idx].set_xlabel('z (Input)', fontsize=10)
    axes[1, idx].set_ylabel("f'(z) (Gradient)", fontsize=10)
    axes[1, idx].set_title(f'{name} Derivative', fontsize=11, fontweight='bold')
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_ylim(-0.2, 1.2)

plt.tight_layout()
plt.savefig('lab2_1_activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Key Observations:")
print("-" * 60)
print("1. Sigmoid: Smooth transition, but gradients vanish for |z| > 3")
print("2. Tanh: Similar to sigmoid but zero-centered and steeper")
print("3. ReLU: Constant gradient for z > 0, zero for z < 0")
print("4. Leaky ReLU: Small gradient for z < 0, preventing 'dead neurons'")
```

**Key Insights:**
- **Sigmoid/Tanh:** Smooth phase transitions but gradients die far from origin
- **ReLU:** Sharp transition but maintains gradient for positive inputs
- **Leaky ReLU:** Best of both worlds—maintains some gradient everywhere

---

## Lab 2.2: The Logic Gate Experiment

Can a single neuron "think"? Let's teach one neuron to perform Boolean logic.

### Part A: The AND Gate

Truth table for AND:

| x₁ | x₂ | y |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |

```python
"""
Lab 2.2A: Teaching a Neuron to Compute AND
Can a single perceptron learn Boolean logic?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Truth Table
X = torch.tensor([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.0, 1.0]])
Y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

# 2. Build the Model
# A single neuron: 2 inputs → 1 output with sigmoid activation
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# 3. Training Setup
optimizer = optim.SGD(model.parameters(), lr=1.0)
loss_fn = nn.BCELoss()  # Binary Cross-Entropy for classification

# Training history
losses = []

print("=" * 60)
print("TRAINING: AND Gate")
print("=" * 60)

# 4. Training Loop
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")

# 5. Test the trained model
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
with torch.no_grad():
    predictions = model(X)
    print("\nInput  | Target | Prediction | Rounded")
    print("-" * 45)
    for i in range(len(X)):
        print(f"{X[i].numpy()} | {Y[i].item():.0f}      | {predictions[i].item():.4f}     | {predictions[i].round().item():.0f}")

# 6. Visualization: Decision Boundary
# Extract learned weights and bias
linear_layer = model[0]
w1, w2 = linear_layer.weight[0].detach().numpy()
b = linear_layer.bias[0].detach().numpy()

print(f"\nLearned Parameters:")
print(f"  w₁ = {w1:.4f}, w₂ = {w2:.4f}, b = {b:.4f}")
print(f"  Decision boundary: {w1:.4f}·x₁ + {w2:.4f}·x₂ + {b:.4f} = 0")

# Plot decision boundary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), 
                     np.linspace(-0.5, 1.5, 200))
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = model(grid_points).reshape(xx.shape)

contour = ax1.contourf(xx, yy, Z.numpy(), levels=20, cmap='RdYlBu', alpha=0.6)
ax1.contour(xx, yy, Z.numpy(), levels=[0.5], colors='black', linewidths=2)

# Plot data points
colors = ['red' if y == 0 else 'blue' for y in Y.numpy()]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_title('AND Gate: Decision Boundary', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax1, label='Output Probability')

# Plot 2: Loss convergence
ax2.plot(losses, linewidth=2, color='blue')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (BCE)', fontsize=12)
ax2.set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('lab2_2_and_gate.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Analysis:**
- The neuron successfully learns AND logic
- The decision boundary is a straight line separating (1,1) from the rest
- Loss decreases exponentially, reaching near-zero

---

## Lab 2.3: The XOR Wall—When One Neuron Isn't Enough

Now let's try XOR (Exclusive OR):

| x₁ | x₂ | y |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

```python
"""
Lab 2.3: The XOR Problem
Demonstrating the fundamental limitation of single neurons
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# XOR Truth Table
X = torch.tensor([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Single neuron model (WILL FAIL)
model_single = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model_single.parameters(), lr=1.0)
loss_fn = nn.BCELoss()

losses_single = []

print("=" * 60)
print("ATTEMPTING: XOR with Single Neuron")
print("=" * 60)

for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model_single(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    losses_single.append(loss.item())
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")

print("\n" + "=" * 60)
print("RESULTS (Single Neuron)")
print("=" * 60)
with torch.no_grad():
    predictions = model_single(X)
    print("\nInput  | Target | Prediction | Rounded")
    print("-" * 45)
    for i in range(len(X)):
        print(f"{X[i].numpy()} | {Y[i].item():.0f}      | {predictions[i].item():.4f}     | {predictions[i].round().item():.0f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Failed decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), 
                     np.linspace(-0.5, 1.5, 200))
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = model_single(grid_points).reshape(xx.shape)

contour = axes[0].contourf(xx, yy, Z.numpy(), levels=20, cmap='RdYlBu', alpha=0.6)
axes[0].contour(xx, yy, Z.numpy(), levels=[0.5], colors='black', linewidths=2)

# Plot XOR data points
colors = ['red' if y == 0 else 'blue' for y in Y.numpy()]
axes[0].scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)

# Add labels
for i in range(len(X)):
    axes[0].text(X[i, 0] + 0.05, X[i, 1] + 0.05, f"({int(X[i,0])},{int(X[i,1])})", fontsize=10)

axes[0].set_xlabel('x₁', fontsize=12)
axes[0].set_ylabel('x₂', fontsize=12)
axes[0].set_title('XOR: Impossible for Single Neuron', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
plt.colorbar(contour, ax=axes[0], label='Output Probability')

# Add annotation
axes[0].text(0.5, -0.3, 'No single line can separate blue from red!', 
             ha='center', fontsize=11, color='red', fontweight='bold')

# Plot 2: Loss curve showing failure to converge
axes[1].plot(losses_single, linewidth=2, color='red')
axes[1].axhline(y=0.693, color='black', linestyle='--', linewidth=2, 
                label='Random Guessing (0.693)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (BCE)', fontsize=12)
axes[1].set_title('Loss Stuck Near Random Guessing', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab2_3_xor_failure.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("CONCLUSION: A single neuron CANNOT learn XOR.")
print("The data is NOT linearly separable.")
print("Solution: We need MULTIPLE neurons (layers).")
print("=" * 60)
```

**The Historic Failure:**

This limitation nearly killed neural networks in the 1960s. In 1969, Marvin Minsky and Seymour Papert published *Perceptrons*, mathematically proving that single-layer perceptrons cannot solve XOR.

**THEORY LAB: Linear Separability**

A dataset is **linearly separable** if you can draw a straight line (or hyperplane in higher dimensions) that perfectly separates the classes.

- AND, OR, NAND: Linearly separable ✓
- XOR: NOT linearly separable ✗

**The XOR problem requires two lines:**
- One line to separate (0,0) and (1,1) from the others
- But you can't do it with a single line!

**Solution:** Stack neurons into layers. This is the birth of **Multi-Layer Perceptrons (MLPs)**, which we'll build in Part II.

---

## Lab 2.4: Multi-Neuron Solution to XOR

Let's preview the solution using a 2-layer network:

```python
"""
Lab 2.4: Solving XOR with Multiple Neurons
Demonstrating the power of hidden layers
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# XOR data
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Multi-layer model: 2 inputs → 4 hidden neurons → 1 output
model_mlp = nn.Sequential(
    nn.Linear(2, 4),      # Hidden layer with 4 neurons
    nn.Sigmoid(),
    nn.Linear(4, 1),      # Output layer
    nn.Sigmoid()
)

optimizer = optim.Adam(model_mlp.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

losses_mlp = []

print("=" * 60)
print("TRAINING: XOR with Multi-Layer Perceptron")
print("=" * 60)

for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model_mlp(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    losses_mlp.append(loss.item())
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")

print("\n" + "=" * 60)
print("RESULTS (Multi-Layer Network)")
print("=" * 60)
with torch.no_grad():
    predictions = model_mlp(X)
    print("\nInput  | Target | Prediction | Rounded")
    print("-" * 45)
    for i in range(len(X)):
        print(f"{X[i].numpy()} | {Y[i].item():.0f}      | {predictions[i].item():.4f}     | {predictions[i].round().item():.0f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Successful decision boundary
import numpy as np
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), 
                     np.linspace(-0.5, 1.5, 200))
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = model_mlp(grid_points).reshape(xx.shape)

contour = ax1.contourf(xx, yy, Z.numpy(), levels=20, cmap='RdYlBu', alpha=0.6)
ax1.contour(xx, yy, Z.numpy(), levels=[0.5], colors='black', linewidths=3)

colors = ['red' if y == 0 else 'blue' for y in Y.numpy()]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)

ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_title('XOR: SOLVED with Hidden Layer!', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax1, label='Output Probability')

# Plot 2: Loss comparison
ax2.plot(losses_single[:2000], linewidth=2, color='red', label='Single Neuron (Failed)', alpha=0.7)
ax2.plot(losses_mlp, linewidth=2, color='green', label='Multi-Layer (Success)')
ax2.axhline(y=0.693, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (BCE)', fontsize=12)
ax2.set_title('Loss Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('lab2_4_xor_solved.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("SUCCESS: Multi-layer network learns XOR perfectly!")
print("Hidden neurons create NONLINEAR decision boundaries.")
print("=" * 60)
```

**Key Insight:** The hidden layer transforms the input space, making the data linearly separable in a higher-dimensional space. This is the fundamental principle behind deep learning.

---

## Real-World Application: Binary Classification

Let's apply our neuron to a real-world problem: classifying whether a person has diabetes based on medical measurements.

```python
"""
Lab 2.5: Real-World Binary Classification
Predicting diabetes from medical features
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load diabetes dataset (regression) and convert to binary classification
# We'll predict if progression is above median (binary problem)
diabetes = load_diabetes()
X = diabetes.data
y = (diabetes.target > np.median(diabetes.target)).astype(float).reshape(-1, 1)

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Build model: Simple logistic regression (single neuron)
model = nn.Sequential(
    nn.Linear(10, 1),  # 10 medical features → 1 output
    nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Training
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(500):
    # Training
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_t)
    loss_train = loss_fn(y_pred_train, y_train_t)
    loss_train.backward()
    optimizer.step()
    
    train_losses.append(loss_train.item())
    
    # Testing
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        loss_test = loss_fn(y_pred_test, y_test_t)
        accuracy = ((y_pred_test > 0.5).float() == y_test_t).float().mean()
        
        test_losses.append(loss_test.item())
        test_accuracies.append(accuracy.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss = {loss_train.item():.4f}, "
              f"Test Loss = {loss_test.item():.4f}, Test Acc = {accuracy.item():.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Loss curves
axes[0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss (BCE)', fontsize=11)
axes[0].set_title('Learning Curves', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[1].plot(test_accuracies, color='green', linewidth=2)
axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random Guess')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Test Accuracy', fontsize=11)
axes[1].set_title('Model Performance', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Predictions vs True Labels
with torch.no_grad():
    y_pred_final = model(X_test_t).numpy().flatten()

axes[2].scatter(range(len(y_test)), y_test, alpha=0.5, label='True Labels', s=50)
axes[2].scatter(range(len(y_pred_final)), y_pred_final, alpha=0.5, label='Predictions', s=30)
axes[2].axhline(y=0.5, color='black', linestyle='--', linewidth=2)
axes[2].set_xlabel('Sample Index', fontsize=11)
axes[2].set_ylabel('Diabetes Risk', fontsize=11)
axes[2].set_title('Predictions vs Ground Truth', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab2_5_diabetes_classification.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.2%}")
```

**Why This Matters in Production:**
- Logistic regression (single neuron) is still widely used in medicine, finance, and marketing
- It's interpretable: you can explain which features contribute to predictions
- It's fast and requires minimal data compared to deep networks

---

## Exercises

### Exercise 2.1: OR and NAND Gates
Train single neurons to learn:
1. OR gate
2. NAND gate

Plot their decision boundaries.

### Exercise 2.2: Activation Function Impact
Retrain the AND gate using:
1. Tanh activation
2. ReLU activation

Does it still converge? Why or why not?

### Exercise 2.3: Feature Importance
In the diabetes example, extract the weights from the trained model. Which medical feature has the strongest positive/negative influence on the prediction?

### Exercise 2.4: Custom Activation
Implement a custom activation function:

**f(z) = z/√(1 + z²)**  (Softplus variant)

Use it in the AND gate experiment. Does it work?

---

## Exercise Solutions

### Solution 2.1: OR Gate

```python
import torch
import torch.nn as nn
import torch.optim as optim

# OR Truth Table
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_or = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
optimizer = optim.SGD(model.parameters(), lr=1.0)
loss_fn = nn.BCELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    loss = loss_fn(model(X), Y_or)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    print("OR Gate Results:")
    print(model(X).round())
```

### Solution 2.3: Feature Importance (Diabetes)

```python
# Extract weights from trained model
weights = model[0].weight.detach().numpy().flatten()
feature_names = diabetes.feature_names

# Sort by absolute magnitude
importance = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)

print("Feature Importance (Diabetes Prediction):")
for name, weight in importance:
    direction = "↑ increases" if weight > 0 else "↓ decreases"
    print(f"{name:6s}: {weight:+.4f} {direction} risk")
```

---

## Chapter Summary

In this chapter, you've learned:

1. **Neurons are Weighted Summations:** They compute w^T·x + b and apply nonlinearity

2. **Activation Functions Create Intelligence:** Linear systems cannot learn complex patterns; nonlinearity is essential

3. **Single Neurons Have Limits:** They can only separate linearly separable data

4. **The XOR Problem:** Historically proved that single-layer networks are insufficient

5. **Hidden Layers Unlock Power:** Multiple neurons can create arbitrarily complex decision boundaries

**REMEMBER:** A single neuron is a linear classifier with a nonlinear decision boundary (due to activation). To solve complex problems like XOR, we need depth—multiple layers transforming the input space.

---

## Looking Ahead

We've mastered the atom. In **Chapter 3**, we'll study the force that moves these atoms through parameter space: **Gradient Descent** in high dimensions. We'll learn about backpropagation, the chain rule of calculus that allows us to train networks with millions of parameters.

Then in **Part II**, we'll assemble these atoms into molecules—full Neural Networks capable of seeing, reading, and reasoning.
