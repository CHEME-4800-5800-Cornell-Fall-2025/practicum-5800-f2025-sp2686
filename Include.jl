# setup paths -
# setup paths (keep same initials) -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_IMAGES = joinpath(_ROOT, "images-uncorrelated");

# load package manager and activate/instantiate if needed
using Pkg
if !isfile(joinpath(_ROOT, "Manifest.toml"))
    # activate current project and ensure environment is instantiated
    Pkg.activate(_ROOT)
    try
        Pkg.instantiate()
    catch err
        @warn "Pkg.instantiate() failed; continuing. Error: $err"
    end
end

# List of packages required by the notebook (match the original `using` list)
const _REQUIRED_PKGS = [
    "Images",
    "ImageInTerminal",
    "FileIO",
    "ImageIO",
    "OneHotArrays",
    "Statistics",
    "JLD2",
    "LinearAlgebra",
    "Plots",
    "Colors",
    "Distances",
    "DataStructures",
    "Test",
    "IJulia",
]

# ensure packages are installed and then load them
function _ensure_and_load(pkgs)
    for p in pkgs
        try
            @eval using $(Symbol(p))
        catch err
            @info "Package $p not found; adding via Pkg.add" error=err
            try
                Pkg.add(p)
                @eval using $(Symbol(p))
            catch err2
                @error "Failed to add or load package $p" error=err2
                rethrow(err2)
            end
        end
    end
end

_ensure_and_load(_REQUIRED_PKGS)

function decode(s::AbstractVector{<:Integer}; nrows::Int=28, ncols::Int=28)
    @assert length(s) == nrows * ncols "decode: length(s) != nrows * ncols"
    # convert +1 -> 1.0, -1 -> 0.0
    vals = Float32.(s .== 1)
    # original vectors were filled in row-major order; reconstruct as nrows×ncols
    mat = reshape(vals, ncols, nrows)' 
    return mat
end


# load project source files in order: Types first, bring exported name into Main, then others
types_file = joinpath(_PATH_TO_SRC, "Types.jl")
if isfile(types_file)
    include(types_file)
    # Avoid a conflicting `using` import into Main; create a safe alias only if not already defined.
    try
        if !isdefined(Main, :MyClassicalHopfieldNetworkModel)
            @eval Main begin
                const MyClassicalHopfieldNetworkModel = Types.MyClassicalHopfieldNetworkModel
            end
        else
            @info "MyClassicalHopfieldNetworkModel already defined in Main; skipping alias."
        end
    catch err
        @warn "Failed to alias MyClassicalHopfieldNetworkModel" error=(err, catch_backtrace())
    end
else
    @info("Types.jl not found in src/ — skipping include.")
end


