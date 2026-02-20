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

[iter: 1]       Loss: 20.79292          Training time: 0.4714 sec
[iter: 50]      Loss: 2.8525753         Training time: 0.5266 sec
[iter: 100]     Loss: 0.08090146        Training time: 0.5761 sec
[iter: 150]     Loss: 0.01038435        Training time: 0.6252 sec
[iter: 200]     Loss: 0.00826358        Training time: 0.6739 sec
[iter: 250]     Loss: 0.00735587        Training time: 0.7244 sec
[iter: 300]     Loss: 0.00543709        Training time: 0.7746 sec
[iter: 350]     Loss: 0.00601879        Training time: 0.8237 sec
[iter: 400]     Loss: 0.00472129        Training time: 0.874 sec
[iter: 450]     Loss: 0.00574292        Training time: 0.923 sec
[iter: 500]     Loss: 0.00456861        Training time: 0.9943 sec
[iter: 500]     Loss: 0.00393435        Training time: 0.9996 sec

training completed.

hamiltonian: 1.8333362
```


## DiffEqFlux.jl ([HNN in DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/dev/examples/hamiltonian_nn/#Hamiltonian-Neural-Network))
<img width="600" height="400" alt="DiffEqFlux_result" src="https://github.com/user-attachments/assets/ba3e398c-1443-49ef-ae46-2cc858640ada" />
