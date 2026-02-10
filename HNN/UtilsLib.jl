module UtilsLib
    using JLD2, CairoMakie

    # save trained model
    function save_model(fdir::String, fname::String, tstate::Training.TrainState)
        isdir(fdir) || mkpath(fdir)
        fpath = joinpath(fdir, "$(fname).jld2")
        
        jldsave(fpath; tstate=tstate)
    end


    # load trained model
    function load_model(fdir::String, fname::String)
        fpath = joinpath(fdir, "$(fname).jld2")
        return load(fpath, "tstate")
    end

    # save figure
    function save_fig(fdir::String, fname::String, fig)
        isdir(fdir) || mkpath(fdir)
        fpath = joinpath(fdir, "$(fname).pdf")

        CairoMakie.save(fpath, fig)
    end

end # module UtilsLib
