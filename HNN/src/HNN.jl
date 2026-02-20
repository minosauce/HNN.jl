module HNN
    using Lux, LuxCore, MLUtils, Zygote, ForwardDiff, ChainRulesCore,
    ComponentArrays, Random, Statistics, LinearAlgebra, OrdinaryDiffEq

    import ADTypes

    # Hamiltonian Neural Networks (HNNs)
    # Neural networks that learn "the Hamiltonian" of a system from data

    # 1. HNN's state is canonical coordinates and canonical momenta
    # x = [q; p] ∈ R^{2n}
    #
    # Example: 1D Harmonic Oscillator
    # H(x) = 1/2 * k * q^2 + 1/2 * m * p^2 
    #
    # 2. The HNN learns an approximation to the Hamiltonian:
    # f_θ(x) ~= H(x)
    #
    # 3. The differential equations is given by Hamilton's equations:
    # ẋ = J ∇H(x)
    #
    # 4. Loss function:
    # Loss = ||ẋ_pred - ẋ_data||² 
    # 

    # Reference (Boltz.jl) :
    # https://github.com/LuxDL/Boltz.jl/blob/ce12052770ccebcba34f6f0c84f1ba484bcb9b7a/src/layers/hamiltonian.jl#L1-L39


    # struct of the Hamiltonian Neural Network 
    struct HamiltonianNN{
            M   <: Lux.AbstractLuxLayer, 
            T   <: AbstractMatrix{<:Real},
            AD  <: ADTypes.AbstractADType
        } <: Lux.AbstractLuxWrapperLayer{:layer}

        layer   ::M     # hamiltonian neural network (with StatefulLuxLayer)
        J       ::T     # canonical or non-canonical symplectic matrix
        ad      ::AD    # AutoDiff backend for gradient computation
    end


    function HamiltonianNN(layer::M, ad::AD) where {M<: Lux.AbstractLuxLayer, AD<: ADTypes.AbstractADType}
        input_dim = first(values(layer.layers)).in_dims
        J = symplectic_matrix(div(input_dim, 2))

        return HamiltonianNN(layer, J, ad)
    end

    function symplectic_matrix(n::Int)
        # x ∈ R^{2n} (hamiltonian system with n degrees of freedom)
        In = Matrix{Float32}(I, n, n)
        O  = zeros(Float32, n, n)

        return [O In;
                -In O]
    end

    function LuxCore.initialstates(rng::AbstractRNG, hnn::HamiltonianNN)
        return (; layer=LuxCore.initialstates(rng, hnn.layer))
    end


    # 1. Hamiltonian, H(x) 
    function hamiltonian(hnn::HamiltonianNN, x, ps, st::NamedTuple)
        model = StatefulLuxLayer{true}(hnn.layer, ps, st.layer)
        H = sum(model(x, ps))
        return H
    end

    # 2. ∇H(x) = dH/dx
    function grad_HNN(ad::ADTypes.AutoForwardDiff, model, x)
        return ForwardDiff.gradient(sum ∘ model, x) 
    end

    function grad_HNN(ad::ADTypes.AutoZygote, model, x)
        return only(Zygote.gradient(sum ∘ model, x))
    end


    # 3. ẋ = J ∇H(x)
    function (hnn::HamiltonianNN)(x::AbstractVector, ps, st::NamedTuple)
        H, st = hnn(reshape(x, :, 1), ps, st)
        return vec(H), st
    end

    function (hnn::HamiltonianNN)(x::AbstractArray{T,N}, ps, st::NamedTuple) where {T,N}
        model = StatefulLuxLayer{true}(hnn.layer, ps, st.layer)

        ∇H = grad_HNN(hnn.ad, model, x)
        xdot = hnn.J * ∇H
        return xdot, (; layer=model.st)
    end



    # logging callback
    # for Optimization.jl solve()
    function callback_wrapper(interval::Int=0)
        stime = time()

        function callback(state, loss)
            iter = state.iter
            etime = time() - stime

            if interval != 0 && 
                (iter == 1 || iter % interval == 0)
                println(
                    "[iter: $(iter)]\t" *
                    "Loss: $(round(loss, digits=8))\t" *
                    "Training time: $(round(etime, digits=4)) sec"
                )
            end
            return false
        end

        return callback
    end


    # ODE right-hand side for HamiltonianNN
    function hnn_rhs!(dx, x, p, t)
        model, ps_trained = p
        dx .= model(x, ps_trained)
    end

    function NeuralODEProblem(model::StatefulLuxLayer, ps_trained; tspan)
        input_dim = first(values(model.model.layer.layers)).in_dims

        x0 = zeros(Float32, input_dim)
        p = (model, ps_trained)
        prob = ODEProblem(hnn_rhs!, x0, tspan, p)
        return prob
    end

    function rollout(prob, x0; solver, saveat  = nothing)
        prob2 = remake(prob, u0=x0)
        sol = solve(prob2, solver; 
            saveat = isnothing(saveat) ? nothing : saveat
        )

        return Array(sol)
    end

end # module HNN
