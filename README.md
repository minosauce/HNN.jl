# HNN.jl

**Hamiltonian Neural Network implemented with Lux.jl**

Hamiltonian Neural Networks (HNNs) are neural networks that learn the **Hamiltonian function** of a dynamical system directly from data, while preserving the underlying **symplectic structure**.

---

## 1. State Representation

The system state is defined in **canonical coordinates** and **canonical momenta**:

\[
x = \begin{bmatrix} q \\ p \end{bmatrix} \in \mathbb{R}^{2n}
\]

The canonical symplectic matrix is given by:

\[
J =
\begin{bmatrix}
0 & I \\
- I & 0
\end{bmatrix}
\]

---

## 2. Example: 1D Spring–Mass System

The Hamiltonian of a one-dimensional spring–mass system is

\[
H(x) = \frac{1}{2} k q^2 + \frac{1}{2} m p^2
\]

---

## 3. Hamiltonian Approximation

The neural network learns an approximation of the Hamiltonian:

\[
f_\theta(x) \approx H(x)
\]

---

## 4. Dynamics (Hamilton’s Equations)

The system dynamics are governed by Hamilton’s equations:

\[
\dot{x} = J \nabla H(x)
\]

---

## 5. Training Objective

The model is trained by minimizing the discrepancy between predicted and true state derivatives:

\[
\mathcal{L} = \left\| \dot{x}_{\text{pred}} - \dot{x}_{\text{data}} \right\|^2
\]


## Custom HNN.jl 
<img width="600" height="400" alt="HNN_results" src="https://github.com/user-attachments/assets/03803c8d-34c3-4340-8497-8767d30852d1" />

### Computing Env
- Mac mini m4

### Training Log
```text
training started...

(epoch : 1 / 125)       Training time: 2.6261 [sec]     Loss: 42.271034
(epoch : 20 / 125)      Training time: 21.4847 [sec]    Loss: 0.3660409
(epoch : 40 / 125)      Training time: 41.6146 [sec]    Loss: 0.024153993
(epoch : 60 / 125)      Training time: 61.5919 [sec]    Loss: 0.015001616
(epoch : 80 / 125)      Training time: 81.5532 [sec]    Loss: 0.011885091
(epoch : 100 / 125)     Training time: 101.5177 [sec]   Loss: 0.010895904
(epoch : 120 / 125)     Training time: 121.5637 [sec]   Loss: 0.010584581
(epoch : 125 / 125)     Training time: 126.5841 [sec]   Loss: 0.010543893

training completed.
```


## DiffEqFlux.jl ([HNN in DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/dev/examples/hamiltonian_nn/#Hamiltonian-Neural-Network))
<img width="600" height="400" alt="DiffEqFlux_result" src="https://github.com/user-attachments/assets/ba3e398c-1443-49ef-ae46-2cc858640ada" />
