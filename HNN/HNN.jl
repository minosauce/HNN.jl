module HNN
    using Lux, MLUtils, Zygote, ForwardDiff,
    ComponentArrays, Random, Statistics, LinearAlgebra, OrdinaryDiffEq

    import ADTypes
    import DifferentiationInterface as DI

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

    function canonical_symplectic(n::Int)
        # x ∈ R^{2n} (hamiltonian system with n degrees of freedom)
        In = Matrix{Float32}(I, n, n)
        O  = zeros(Float32, n, n)

        return [O In;
                -In O]
    end


    # 1. Hamiltonian, H(x) 
    # H(x): vector -> scalar
    @inline function hamiltonian(hnn::HamiltonianNN, x::AbstractVector, ps, st::NamedTuple)
        H, _ = Lux.apply(hnn.layer, x, ps, st)
        return only(H)
    end

    # H(X): matrix(n×B) -> (1×B) or (B)
    @inline function hamiltonian(hnn::HamiltonianNN, X::AbstractMatrix, ps, st::NamedTuple)
        H, _ = Lux.apply(hnn.layer, X, ps, st)
        return H
    end

    # 2. ∇H(x) = dH/dx
    # ∇H(x): vector -> vector
    @inline function grad_HNN(hnn::HamiltonianNN, x::AbstractVector, ps, st::NamedTuple)
        f(u) = hamiltonian(hnn, u, ps, st)
        ∇H = DI.gradient(f, hnn.ad, x)
        return ∇H
    end

    # # matrix(n×B) -> matrix(n×B)
    # function grad_HNN(hnn::HamiltonianNN, X::AbstractMatrix, ps, st::NamedTuple)
    #     function summed_H(U)
    #         H = hamiltonian(hnn, U, ps, st)   # (1,B) or (B,)
    #         return sum(H)                     # scalar
    #     end
        
    #     ∇H = DI.gradient(summed_H, hnn.ad, X)   # (n × B)
    #     return ∇H
    # end

    function grad_HNN(hnn::HamiltonianNN, X::AbstractMatrix, ps, st::NamedTuple)
        f(u) = hamiltonian(hnn, u, ps, st)
        ∇H = @view Lux.batched_jacobian(f, hnn.ad, X)[1, :, :]
        return ∇H
    end

    # 3. ẋ = J ∇H(x) 
    # Lux.apply overloads
    @inline function Lux.apply(hnn::HamiltonianNN, x::AbstractVecOrMat, ps, st::NamedTuple)
        ∇H = grad_HNN(hnn, x, ps, st)        
        xdot = hnn.J * ∇H
        return xdot, st
    end

    # logging callback
    # for Optimization.jl solve()
    function callback_wrapper(interval::Int=0)
        stime = time()

        function callback(state, loss)
            iter = state.iter
            etime = time() - stime

            if !(interval==0) && 
                (iter == 1 || iter % interval == 0)
                println(
                    "(iter : $(iter))\t" *
                    "Loss: $(loss)\t" *
                    "Training time: $(round(etime, digits=4)) [sec]"
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


    # Wrapper for ODE solver
    function rollout(
            trained,
            model,
            x0;
            tspan, 
            solver,
            saveat  = nothing
        )

        # ODE input, extract trained model parameters
        # trained.u = ps (neural parameters) 
        ps_trained = trained.u
        p = (model, ps_trained)

        prob = ODEProblem(hnn_rhs!, x0, tspan, p)

        sol = solve(prob, solver; 
            saveat = isnothing(saveat) ? nothing : saveat
        )

        return Array(sol)
    end

end # module HNN
