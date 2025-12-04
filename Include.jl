##############  SETUP PATHS  ##############
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src")
const _PATH_TO_IMAGES = joinpath(_ROOT, "images-uncorrelated")

##############  PACKAGE SETUP  ##############
using Pkg

# Activate environment
Pkg.activate(_ROOT)
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate failed" error=err
end

# Required packages
const _REQUIRED_PKGS = [
    "Images", "ImageInTerminal", "FileIO", "ImageIO",
    "OneHotArrays", "Statistics", "JLD2", "LinearAlgebra",
    "Plots", "Colors", "Distances", "DataStructures",
    "Test", "IJulia"
]

# Ensure and load packages
function _ensure_and_load(pkgs)
    for p in pkgs
        try
            @eval using $(Symbol(p))
        catch
            @info "Installing package $p …"
            Pkg.add(p)
            @eval using $(Symbol(p))
        end
    end
end

_ensure_and_load(_REQUIRED_PKGS)

##############  LOAD Types.jl ##############
types_file = joinpath(_PATH_TO_SRC, "Types.jl")
if isfile(types_file)
    include(types_file)
    @info "Loaded Types.jl"

    # Bring model type into Main
    if !isdefined(Main, :MyClassicalHopfieldNetworkModel)
        @eval Main const MyClassicalHopfieldNetworkModel = Types.MyClassicalHopfieldNetworkModel
    end
else
    @error "Types.jl not found at $types_file"
end

##############  LOAD factory.jl (defines build) ##############
factory_file = joinpath(_PATH_TO_SRC, "factory.jl")
if isfile(factory_file)
    include(factory_file)
    @info "Loaded factory.jl (build() is now defined)"
else
    @error "factory.jl not found at $factory_file"
end

##############  LOAD Compute.jl (defines recover) ##############
compute_file = joinpath(_PATH_TO_SRC, "Compute.jl")
if isfile(compute_file)
    include(compute_file)
    @info "Loaded Compute.jl (recover() is now defined)"
else
    @error "Compute.jl not found at $compute_file"
end

##############  DECODE FUNCTION ##############
function decode(s::AbstractVector{<:Integer}; nrows::Int=28, ncols::Int=28)
    @assert length(s) == nrows*ncols "decode: wrong vector length"
    vals = Float32.(s .== 1)
    return reshape(vals, ncols, nrows)'
end

##############  BUILD MODEL HELPER FUNCTION ##############
function build_hopfield_model(image_index_set_to_encode, training_image_dataset, number_of_pixels)
    num_images = length(image_index_set_to_encode)
    linearimagecollection = Array{Int32,2}(undef, number_of_pixels, num_images)

    index_vector = sort(collect(image_index_set_to_encode))

    for k ∈ eachindex(index_vector)
        j = index_vector[k]
        ŝₖ = training_image_dataset[j]

        for i ∈ 1:number_of_pixels
            pixel = round(Int, ŝₖ[i])
            linearimagecollection[i,k] = pixel == 0 ? -1 : 1
        end
    end

    # Use factory.jl build() function
    return build(MyClassicalHopfieldNetworkModel; memories=linearimagecollection)
end
