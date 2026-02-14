module LuxHNN
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
        } <: Lux.AbstractLuxWrapperLayer{:model}

        model   ::M     # hamiltonian neural network
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
    @inline function hamiltonian(hnn::HamiltonianNN, x::AbstractVector, ps, st::NamedTuple)
        H, _ = hnn.model(x, ps, st)
        return only(H)
    end

    # 2. ∇H(x) = dH/dx
    # common gradient function
    function grad_HNN(hnn::HamiltonianNN, x::AbstractVector, ps, st::NamedTuple)
        f = u -> hamiltonian(hnn, u, ps, st)
        ∇H = DI.gradient(f, hnn.ad, x)
        return ∇H
    end

    # 3. ẋ = J ∇H(x) (call overloading)
    # vector
    function (hnn::HamiltonianNN)(x::AbstractVector, ps, st)
        ∇H = grad_HNN(hnn, x, ps, st)
        xdot = hnn.J * ∇H
        return xdot, st
    end


    # 4. Loss function for HNN
    function loss_fn(model, ps, st, databatch)
        X, Xdot = databatch
        batchsize = size(X, 2)

        loss = mapreduce(+, 1:batchsize) do k
            x = @view X[:, k]
            xdot_true = @view Xdot[:, k]
            xdot_pred, _ = model(x, ps, st)
            sum(abs2, xdot_pred .- xdot_true)
        end

        return loss / batchsize, st, NamedTuple()
    end


    # Training loop for HNN
    function train!(
            tstate::Training.TrainState,
            dataloader::DataLoader;
            epochs::Int = 100,
            log_interval::Int = 0,
            ad::ADTypes.AbstractADType = AutoForwardDiff()
        )

        etime = 0f0 # epoch time

        for epoch in 1:epochs
            epoch_loss = 0f0
            nbatch = 0

            stime = time() # start time
            for databatch in dataloader
                # compute loss
                _, loss, _, tstate = Training.single_train_step!(
                    ad, loss_fn, databatch, tstate)

                epoch_loss += loss
                nbatch += 1
            end
            ttime = time() - stime # terminal time
            etime += ttime

            # average loss over batches
            epoch_loss /= nbatch
            
            # logging
            if !(log_interval==0) && 
                (epoch == 1 || epoch % log_interval == 0 || epoch == epochs)

                callback(epoch, epochs, etime, epoch_loss)
            end

        end # epoch loop

        return tstate
    end

    # logging callback
    function callback(epoch::Int, epochs::Int, etime, epoch_loss)
        println(
            "(epoch : $epoch / $epochs)\t" *
            "Loss: $(epoch_loss)\t" *
            "Training time: $(round(etime, digits=4)) [sec]"
            )
    end


    # ODE right-hand side for HamiltonianNN
    function hnn_rhs!(dx, x, p, t)
        model, ps_trained, st_trained = p
        dx .= model(x, ps_trained, st_trained) |> first
    end


    # Wrapper for ODE solver
    function rollout(
            tstate::Training.TrainState, 
            x0;
            tspan, 
            solver,
            saveat  = nothing
        )

        # ODE input, extract trained model parameters
        model = tstate.model
        ps = tstate.parameters
        st = Lux.testmode(tstate.states)
        p = (model, ps, st)

        prob = ODEProblem(hnn_rhs!, x0, tspan, p)

        sol = solve(prob, solver; 
            saveat = isnothing(saveat) ? nothing : saveat
        )

        return Array(sol)
    end

end # module HNN
