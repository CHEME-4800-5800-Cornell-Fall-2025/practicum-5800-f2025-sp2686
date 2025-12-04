# Minimal compute utilities for the Hopfield practicum

# compute energy for a ±1 integer state vector s
function _energy(W::AbstractMatrix{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat}, s::AbstractVector{<:Integer})
    sf = Float32.(s)
    return -0.5f0 * dot(sf, W * sf) - dot(b, sf)
end

# Hamming distance (also used in notebook)
hamming(a::AbstractVector, b::AbstractVector) = sum(a .!= b)

"""
    recover(model, s0, true_image_energy; maxiterations=1000, patience=5, miniterations_before_convergence=nothing)

Asynchronous Hopfield recovery. Returns (frames, energydictionary).
"""
function recover(model, s0::AbstractVector{<:Integer}, true_image_energy::Real;
                 maxiterations::Int = 1000,
                 patience::Int = 5,
                 miniterations_before_convergence::Union{Int,Nothing} = nothing)

    W = getfield(model, :W)
    b = getfield(model, :b)
    memories = hasfield(typeof(model), :memories) ? getfield(model, :memories) : nothing

    N = size(W, 1)
    @assert length(s0) == N "recover: length(s0) != size(model.W,1)"

    s = Array{Int32,1}(s0) |> copy
    frames = Dict{Int, Array{Int32,1}}()
    energydictionary = Dict{Int, Float32}()

    miniter = miniterations_before_convergence === nothing ? max(patience, 1) : miniterations_before_convergence

    function _matches_any_memory(svec, mems)
        if mems === nothing
            return false
        end
        for k in 1:size(mems, 2)
            if isequal(svec, mems[:, k])
                return true
            end
        end
        return false
    end

    recent = Vector{Array{Int32,1}}()

    for t in 1:maxiterations
        i = rand(1:N)
        hi = dot(W[i, :], Float32.(s)) - b[i]
        s[i] = hi >= 0f0 ? Int32(1) : Int32(-1)

        frames[t] = copy(s)
        energydictionary[t] = Float32(_energy(W, b, s))

        push!(recent, copy(s))
        if length(recent) > patience
            popfirst!(recent)
        end

        if t >= miniter

            # FIXED: guaranteed-bool vector comparison
            identical_recent =
                length(recent) >= patience &&
                all(x -> isequal(x, recent[1]), recent)

            if identical_recent
                return frames, energydictionary
            end

            if _matches_any_memory(s, memories)
                return frames, energydictionary
            end

            if energydictionary[t] <= Float32(true_image_energy)
                return frames, energydictionary
            end
        end
    end

    return frames, energydictionary
end
