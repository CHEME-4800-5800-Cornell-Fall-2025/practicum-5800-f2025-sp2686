module Types

export MyClassicalHopfieldNetworkModel

"""
    MyClassicalHopfieldNetworkModel

Container for a classical Hopfield network.

Fields:
- W::Array{Float32,2}       : weight matrix (N×N), symmetric with zero diagonal
- b::Array{Float32,1}       : bias vector (N), typically zeros
- energy::Dict{Int,Float32} : per-memory energies (keys 1:K)
- memories::Union{Array{Int32,2},Nothing} : stored patterns (N×K) as ±1, or nothing
"""
struct MyClassicalHopfieldNetworkModel
    W::Array{Float32,2}
    b::Array{Float32,1}
    energy::Dict{Int,Float32}
    memories::Union{Array{Int32,2}, Nothing}
end

# Convenience constructor
function MyClassicalHopfieldNetworkModel(W::AbstractMatrix,
                                         b::AbstractVector,
                                         energy::Dict{Int,Float32};
                                         memories=nothing)

    Wf = Float32.(Array(W))
    bf = Float32.(Array(b))
    mem = memories === nothing ? nothing : Array{Int32,2}(memories)

    # enforce zero diagonal
    for i in 1:min(size(Wf,1), size(Wf,2))
        Wf[i,i] = 0f0
    end

    return MyClassicalHopfieldNetworkModel(Wf, bf, energy, mem)
end

# Pretty-print
Base.show(io::IO, m::MyClassicalHopfieldNetworkModel) = print(io,
    "MyClassicalHopfieldNetworkModel(N=$(size(m.W,1)), K=$(m.memories === nothing ? 0 : size(m.memories,2)))"
)

end