# HNN.jl
Hamiltonian Neural Network (HNN by Lux.jl)

<br>
Neural networks that learn "the Hamiltonian" of a system from data
<br>

<br>
1. HNN's state is canonical coordinates and canonical momenta
<br>
x = [q; p] ∈ R^{2n}  % HNN's state
<br>
J = [0 I; -I 0]      % canonical symplectic matrix
<br>

<br>
[Example: 1D Spring mass system]
<br>

<br>
H(x) = 1/2 * k * q^2 + 1/2 * m * p^2
<br>

<br>
2. The HNN learns an approximation to the Hamiltonian:
<br>
f_θ(x) ~= H(x)
<br>

<br>
3. The differential equations is given by Hamilton's equations:
<br>
ẋ = J ∇H(x)
<br>

<br>
4. Loss function:
<br>
Loss = ||ẋ_pred - ẋ_data||²
<br>

## Custom HNN.jl 
<img width="600" height="400" alt="HNN_results" src="https://github.com/user-attachments/assets/03803c8d-34c3-4340-8497-8767d30852d1" />



## DiffEqFlux.jl ([HNN in DiffEqFlux.jl](https://docs.sciml.ai/DiffEqFlux/dev/examples/hamiltonian_nn/#Hamiltonian-Neural-Network))
<img width="600" height="400" alt="DiffEqFlux_result" src="https://github.com/user-attachments/assets/ba3e398c-1443-49ef-ae46-2cc858640ada" />
