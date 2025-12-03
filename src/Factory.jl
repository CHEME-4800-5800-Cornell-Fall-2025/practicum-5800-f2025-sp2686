
# Resolve the MyClassicalHopfieldNetworkModel type (could be in Main or in Types)
function _resolve_model_type()
    if isdefined(Main, :MyClassicalHopfieldNetworkModel)
        return Main.MyClassicalHopfieldNetworkModel
    elseif isdefined(Types, :MyClassicalHopfieldNetworkModel)
        return Types.MyClassicalHopfieldNetworkModel
    else
        throw(ErrorException("MyClassicalHopfieldNetworkModel not found. Include src/Types.jl first."))
    end
end

# Compute Hebbian weights from memories matrix (N × K) with ±1 entries
function _compute_hebbian_weights(mem::AbstractMatrix)
    N, K = size(mem)
    M = Float32.(Array(mem))
    W = zeros(Float32, N, N)
    for k in 1:K
        W .+= M[:,k] * M[:,k]'    # outer product
    end
    W ./= K
    # enforce symmetry and zero diagonal
    W = 0.5f0 * (W + W')
    for i in 1:N
        W[i,i] = 0f0
    end
    return W
end

# Build API: support build(T, NamedTuple) and keyword form build(T; memories=...)
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

function build(::Type{T}; memories) where {T}
    modeltype = _resolve_model_type()
    if T != modeltype
        throw(ArgumentError("build: unsupported type. Expected $(modeltype)."))
    end
    @assert memories !== nothing "build: 'memories' is required"
    mem = Array{Int32,2}(memories)
    W = _compute_hebbian_weights(mem)
    b = zeros(Float32, size(W,1))
    energy = Dict{Int,Float32}()
    for k in 1:size(mem,2)
        sk = Float32.(mem[:,k])
        energy[k] = -0.5f0 * dot(sk, W * sk) - dot(b, sk)
    end
    # Construct model via its constructor (works whether type is in Main or Types)
    return T(W, b, energy; memories = mem)
end

# Friendly fallback to catch the old placeholder call
function build(::Type, args...)
    throw(ArgumentError("build: unsupported call signature. Use `build(MyClassicalHopfieldNetworkModel, (memories=... ,))` or `build(MyClassicalHopfieldNetworkModel; memories=...)`"))
end

