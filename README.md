# Manual Backprop Neural Net: Deconstructing Deep Learning

🌍 [한국어](README.ko.md) | **English**

![image](https://i.imgur.com/qrYfsnh.png)

## 1. Project Overview
> **"What I cannot create, I do not understand."** - Richard Feynman

This project is a deep learning library implemented from scratch using **only `NumPy`**, **without PyTorch's `autograd` engine**, to handle the entire training process (Forward, Backward, Optimizer).
It aims to understand the mathematical essence of Computational Graphs and Backpropagation by reverse-engineering the internal mechanisms of deep learning frameworks, often perceived as "black boxes."

## 2. Key Features
* **Pure NumPy Implementation:** 0% dependency on `torch.autograd`.
* **Modular Design:** Object-oriented design based on `Layer`.
* **Mathematical Rigor:** Precise gradient calculations based on the Chain Rule.

## 3. Mathematical Foundations
This library propagates Local Gradients to the Upstream via the **Chain Rule**.

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

In particular, it demonstrates at the code level that the backpropagation of the `Softmax-with-Loss` layer is elegantly derived as follows:

$$
\frac{\partial L}{\partial z_k} = y_k - t_k
$$
(Where $y_k$ is the softmax output and $t_k$ is the ground truth label.)

## 4. Verification
The correctness of the implementation is rigorously verified in two ways:
1. **Gradient Checking:** Comparison with Numerical Differentiation.
2. **Cross-Validation with PyTorch:** Ensuring consistency with PyTorch's automatic differentiation results within an error margin of $10^{-5}$.