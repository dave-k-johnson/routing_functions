# copied and adapted from https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl/blob/main/src/homotopies/parameter_homotopy.jl

struct RoutingPointsHomotopy <: AbstractHomotopy
    ∇r::RoutingGradient
    p::Vector{ComplexF64}
    q::Vector{ComplexF64}
    #cache
    t_cache::Base.RefValue{ComplexF64}
    pt::Vector{ComplexF64}
    taylor_pt::TaylorVector{2,ComplexF64}
end

function RoutingPointsHomotopy(∇r::RoutingGradient, p::AbstractVector, q::AbstractVector)
    @assert length(p) == length(q) == size(∇r)[1]

    p̂ = Vector{ComplexF64}(p)
    q̂ = Vector{ComplexF64}(q)
    taylor_pt = TaylorVector{2}(ComplexF64, length(q))
    pt = copy(p̂)

    RoutingPointsHomotopy(∇r, p̂, q̂, Ref(complex(NaN)), pt, taylor_pt)
end

Base.size(H::RoutingPointsHomotopy) = size(H.∇r)

import HomotopyContinuation.start_parameters!
import HomotopyContinuation.target_parameters!
import HomotopyContinuation.parameters!

function start_parameters!(H::RoutingPointsHomotopy, p)
    H.p .= p
    # void cache
    H.t_cache[] = NaN
    H
end
function target_parameters!(H::RoutingPointsHomotopy, q)
    H.q .= q
    H.t_cache[] = NaN
    H
end
function parameters!(H::RoutingPointsHomotopy, p, q)
    start_parameters!(H, p)
    target_parameters!(H, q)
end
# function HomotopyContinuation.parameters!(H::RoutingPointsHomotopy, p, q)
#     H.p .= p
#     H.q .= q
#     H.t_cache[] = NaN
#     H
# end

function tp!(H::RoutingPointsHomotopy, t::Union{ComplexF64,Float64})
    t == H.t_cache[] && return H.taylor_pt

    if imag(t) == 0
        let t = real(t)
            @inbounds for i = 1:length(H.taylor_pt)
                ptᵢ = t * H.p[i] + (1.0 - t) * H.q[i]
                H.pt[i] = ptᵢ
                H.taylor_pt[i] = (ptᵢ, H.p[i] - H.q[i])
            end
        end
    else
        @inbounds for i = 1:length(H.taylor_pt)
            ptᵢ = t * H.p[i] + (1.0 - t) * H.q[i]
            H.pt[i] = ptᵢ
            H.taylor_pt[i] = (ptᵢ, H.p[i] - H.q[i])
        end
    end
    H.t_cache[] = t

    H.taylor_pt
end

function ModelKit.evaluate!(u, H::RoutingPointsHomotopy, x, t)
    tp!(H, t)
    evaluate!(u, H.∇r, x, H.pt)
end

function ModelKit.evaluate_and_jacobian!(u, U, H::RoutingPointsHomotopy, x, t)
    tp!(H, t)
    evaluate_and_jacobian!(u, U, H.∇r, x, H.pt)
end

function ModelKit.taylor!(u, v::Val, H::RoutingPointsHomotopy, tx, t)
    taylor!(u, v, H.∇r, tx, tp!(H, t))
    u
end
