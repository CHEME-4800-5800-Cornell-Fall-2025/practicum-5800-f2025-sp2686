
# Resolve the MyClassicalHopfieldNetworkModel type
# -------------------------------------------------------------
function _resolve_model_type()
    if isdefined(Main, :MyClassicalHopfieldNetworkModel)
        return Main.MyClassicalHopfieldNetworkModel
    elseif @isdefined(Types) && isdefined(Types, :MyClassicalHopfieldNetworkModel)
        return Types.MyClassicalHopfieldNetworkModel
    else
        throw(ErrorException("MyClassicalHopfieldNetworkModel not found. Include src/Types.jl first."))
    end
end

# -------------------------------------------------------------
# Compute Hebbian weights
# -------------------------------------------------------------
function _compute_hebbian_weights(mem::AbstractMatrix)
    N, K = size(mem)
    M = Float32.(Array(mem))
    W = zeros(Float32, N, N)

    for k in 1:K
        W .+= M[:,k] * M[:,k]'      # outer product
    end

    W ./= K
    W = 0.5f0 * (W + W')           # symmetrize

    @inbounds for i in 1:N
        W[i,i] = 0f0
    end

    return W
end

# -------------------------------------------------------------
# Build(T, NamedTuple) form
# -------------------------------------------------------------
function build(::Type{T}, params::NamedTuple) where {T}
    modeltype = _resolve_model_type()
    if T != modeltype
        throw(ArgumentError("build: unsupported type. Expected $(modeltype)."))
    end

    if :memories ∉ propertynames(params)
        throw(ArgumentError("build: expected named tuple with key `:memories`"))
    end

    return build(T; memories = params.memories)
end

# -------------------------------------------------------------
# Build(T; memories=...) form
# -------------------------------------------------------------
function build(::Type{T}; memories) where {T}
    modeltype = _resolve_model_type()
    if T != modeltype
        throw(ArgumentError("build: unsupported type. Expected $(modeltype)."))
    end

    @assert memories !== nothing "build: 'memories' is required"

    mem = Array{Int32,2}(memories)
    W = _compute_hebbian_weights(mem)
    b = zeros(Float32, size(W,1))

    # precompute energies of memories
    energy = Dict{Int,Float32}()
    for k in 1:size(mem,2)
        sk = Float32.(mem[:,k])
        energy[k] = -0.5f0 * dot(sk, W * sk) - dot(b, sk)
    end

    return T(W, b, energy; memories = mem)
end

# -------------------------------------------------------------
# Fallback: unsupported build signatures
# -------------------------------------------------------------
function build(::Type, args...)
    throw(ArgumentError(
        "build: unsupported call signature. " *
        "Use `build(MyClassicalHopfieldNetworkModel, (memories=...,))` " *
        "or `build(MyClassicalHopfieldNetworkModel; memories=...)`"
    ))
end