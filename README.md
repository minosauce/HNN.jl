# HNN.jl

**Hamiltonian Neural Network implemented with Lux.jl**

Hamiltonian Neural Networks (HNNs) are neural networks that learn the **Hamiltonian function** of a dynamical system directly from data, while preserving the underlying **symplectic structure**.

---

## 1. State Representation

The system state is defined in **canonical coordinates** and **canonical momenta**:

$$
x =
\begin{bmatrix}
q \\
p
\end{bmatrix}
\in \mathbb{R}^{2n}
$$

The canonical symplectic matrix is given by:

```math
J =
\begin{bmatrix}
0 & I \\
- I & 0
\end{bmatrix}
```

---

## 2. Example: 1D Spring–Mass System

The Hamiltonian of a one-dimensional spring–mass system is

$$
H(x) = \frac{1}{2} k q^2 + \frac{1}{2} m p^2
$$

---

## 3. Hamiltonian Approximation

The neural network learns an approximation of the Hamiltonian:

$$
f_\theta(x) \approx H(x)
$$

---

## 4. Dynamics (Hamilton’s Equations)

The system dynamics are governed by Hamilton’s equations:

$$
\dot{x} = J \nabla H(x)
$$

---

## 5. Training Objective

The model is trained by minimizing the discrepancy between predicted and true state derivatives:

```math
$$
\mathcal{L}
=
\left\|
\dot{x}_{\text{pred}}
-
\dot{x}_{\text{data}}
\right\|^2
$$
```


## Custom HNN.jl 
<img width="600" height="400" alt="HNN_results" src="https://github.com/user-attachments/assets/03803c8d-34c3-4340-8497-8767d30852d1" />

### Computing Env
- Mac mini m4

### Training Log
```text
training started...

[iter: 1]       Loss: 20.795025         Training time: 12.57 sec
[iter: 50]      Loss: 2.8498154         Training time: 12.8454 sec
[iter: 100]     Loss: 0.079303175       Training time: 12.8944 sec
[iter: 150]     Loss: 0.010104474       Training time: 12.9468 sec
[iter: 200]     Loss: 0.008350119       Training time: 12.9982 sec
[iter: 250]     Loss: 0.0072129625      Training time: 13.0468 sec
[iter: 300]     Loss: 0.0055911485      Training time: 13.0994 sec
[iter: 350]     Loss: 0.0059271716      Training time: 13.1518 sec
[iter: 400]     Loss: 0.0048886715      Training time: 13.2019 sec
[iter: 450]     Loss: 0.0056649945      Training time: 13.2546 sec
[iter: 500]     Loss: 0.004729797       Training time: 13.3069 sec
[iter: 500]     Loss: 0.004729797       Training time: 13.3179 sec

training completed.
```


## DiffEqFlux.jl ([HNN in DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/dev/examples/hamiltonian_nn/#Hamiltonian-Neural-Network))
<img width="600" height="400" alt="DiffEqFlux_result" src="https://github.com/user-attachments/assets/ba3e398c-1443-49ef-ae46-2cc858640ada" />
