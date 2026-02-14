include("LuxHNN.jl")

using Lux, MLUtils, ComponentArrays, Random, 
Optimisers, Plots, OrdinaryDiffEq, Statistics,
DiffEqFlux, Optimization, OptimizationOptimisers

import ADTypes

const ad_forward = ADTypes.AutoForwardDiff()
const ad_zygote = ADTypes.AutoZygote()

##
# 1. Data generation: 1D Spring mass system
#
t = range(0.0f0, 1.0f0; length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

# Std of noise
σ_q = 0.01f0
σ_p = 0.02f0

noise = vcat(
    σ_q .* randn(Float32, size(q_t)),
    σ_p .* randn(Float32, size(p_t))
)

data = vcat(q_t, p_t)
data_noise = (vcat(q_t, p_t) + noise)

target = vcat(dqdt, dpdt)
batchsize = 256
epochs = 125 
log_itv = 20

# for clean test
# dataloader = DataLoader((data, target); batchsize = batchsize)

# for noisy test
dataloader = DataLoader((data_noise, target); batchsize = batchsize)


##
# 2. Hamiltonian Neural Network by LuxHNN.jl
#

H_net = Chain(
    Dense(2, 32, gelu),
    Dense(32, 32, gelu),
    Dense(32, 1)
)

J = LuxHNN.canonical_symplectic(1) 

# Hamiltonian Neural Network model
model = LuxHNN.HamiltonianNN(H_net, J, ad_forward)
ps, st = Lux.setup(Xoshiro(0), model) 
ps = ps |> ComponentArray

opt = Optimisers.Adam(0.003f0)
tstate = Training.TrainState(model, ps, st, opt)

println("training started...\n")
tstate = LuxHNN.train!(tstate, 
                    dataloader, 
                    epochs = epochs, 
                    log_interval = log_itv,
                    ad = ad_forward)

println("\ntraining completed.")

# Prediction
pred = LuxHNN.rollout(tstate, data[:, 1], tspan=(0.0f0, 1.0f0), solver=Tsit5(), saveat=t)

# plotting
p1 = plot(data[1, :], data[2, :]; lw = 4, label = "Ground Truth")
plot!(p1, pred[1, :], pred[2, :]; lw = 4, label = "LuxHNN.jl Predicted")
scatter!(p1,
    data_noise[1, :],
    data_noise[2, :];
    markersize = 2,
    alpha = 0.35,
    color = :black,
    label = "Noisy samples"
)
xlabel!("Position (q)")
ylabel!("Momentum (p)")
display(p1)

##
# 3. Hamiltonian Neural Network by DiffEqFlux.jl
#
hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (32, 32, 1), gelu); autodiff = ad_zygote)
ps, st = Lux.setup(Xoshiro(0), hnn)
model = StatefulLuxLayer(hnn, ps, st)
ps_c = ps |> ComponentArray


function loss_function(ps, databatch)
    data, target = databatch
    pred = model(data, ps)
    return mean(abs2, pred .- target)
end

function callback(state, loss)
    println("[Hamiltonian NN] Loss: ", loss)
    return false
end

opt = OptimizationOptimisers.Adam(0.003f0)
opt_func = OptimizationFunction(loss_function, ad_zygote)
opt_prob = OptimizationProblem(opt_func, ps_c, dataloader)

# training
res = @time Optimization.solve(opt_prob, opt; callback, epochs = epochs)

ps_trained = res.u

nhde = NeuralODE(
    hnn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, save_start = true, saveat = t)

# Prediction
pred = Array(first(nhde(data[:, 1], ps_trained, st)))

# plotting
p2 = plot(data[1, :], data[2, :]; lw = 4, label = "Original")
plot!(p2, pred[1, :], pred[2, :]; lw = 4, label = "DiffEqFlux.jl Predicted")
scatter!(p2,
    data_noise[1, :],
    data_noise[2, :];
    markersize = 2,
    alpha = 0.35,
    color = :black,
    label = "Noisy samples"
)
xlabel!("Position (q)")
ylabel!("Momentum (p)")
display(p2)
