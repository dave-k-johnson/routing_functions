# gradient and hessian of the routing function in ℝⁿ. the jacobian of the
# compiled ∇r system is the hessian, so no symbolic differentiation happens here.
function ambient_gradient_hessian!(cache::RoutingCache, point::AbstractVector{Float64})
    HC.evaluate_and_jacobian!(cache.grad_val, cache.Hr_val, cache.r.grad_sys, point)
    return cache.grad_val, cache.Hr_val
end

# same, but handing back arrays the caller owns
function ambient_gradient_hessian(cache::RoutingCache, point::AbstractVector{Float64})
    ∇r, Hr = ambient_gradient_hessian!(cache, point)
    return copy(∇r), copy(Hr)
end

function ambient_gradient_hessian(r::RoutingFunction, point::AbstractVector{Float64})
    n = length(r.vars)
    ∇r = zeros(Float64, n)
    Hr = zeros(Float64, n, n)
    HC.evaluate_and_jacobian!(∇r, Hr, r.grad_sys, point)
    return ∇r, Hr
end

# gradient and hessian of any expression in ℝⁿ
function ambient_gradient_hessian(
    g::Expression,
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    ∇g = HC.differentiate(g, vars)
    grad_sys = HC.InterpretedSystem(System(∇g; variables = vars))
    val = zeros(Float64, length(vars))
    H = zeros(Float64, length(vars), length(vars))
    HC.evaluate_and_jacobian!(val, H, grad_sys, point)
    return val, H
end

# gradient of the routing function at a point of X = V(G), in the tangent basis
function gradient(
    cache::RoutingCache,
    point::Vector{Float64}
    )

    ∇r = reshape(evaluate_grad_r(cache.r, point), 1, :)
    V = LA.nullspace(jacobian_G!(cache, point))
    return ∇r * V
end

gradient(r::RoutingFunction, G::Vector{Expression}, point::Vector{Float64}) =
    gradient(RoutingCache(r, G), point)

# the W matrices of "Smooth Connectivity in Real Algebraic Varieties": the second
# fundamental form data that turns an ambient hessian into the hessian on V(G).
#
# `JG` is k × n, `HG` stacks the n × n hessians of the gᵢ row-block by row-block,
# and both are read, never written, so callers may hand in cache buffers.
function _compute_matrices(
    JG::AbstractMatrix{Float64},
    HG::AbstractMatrix{Float64},
    k::Int64
    )

    # columns form an orthonormal basis of T_x(V(G))
    V = LA.nullspace(JG)
    n, d = size(V)

    # well-constrained linear system for the stacked W's: the first k·d rows say
    # ∑ᵢ JG[j,i]·Wᵢ = -Vᵀ ∇²gⱼ V, the last d² rows say ∑ᵢ V[i,j]·Wᵢ = 0.
    A = zeros(Float64, k * d + d * d, n * d)
    @inbounds for j = 1:k, i = 1:n
        Jji = JG[j, i]
        for t = 1:d
            A[(j-1)*d+t, (i-1)*d+t] = Jji
        end
    end
    @inbounds for j = 1:d, i = 1:n
        Vij = V[i, j]
        for t = 1:d
            A[k*d+(j-1)*d+t, (i-1)*d+t] = Vij
        end
    end

    B = zeros(Float64, k * d + d * d, d)
    VtHV = zeros(Float64, d, d)
    tmp = zeros(Float64, n, d)
    @inbounds for j = 1:k
        Hj = view(HG, (j-1)*n+1:j*n, :)
        LA.mul!(tmp, Hj, V)
        LA.mul!(VtHV, transpose(V), tmp)
        B[(j-1)*d+1:j*d, :] .= .-VtHV
    end

    W = A \ B

    return [W[(i-1)*d+1:i*d, :] for i = 1:n], V
end

function compute_matrices(cache::RoutingCache, point::AbstractVector{Float64})
    jacobian_G!(cache, point)
    HC.jacobian!(cache.HG_val, cache.∇G_sys, point)
    return _compute_matrices(cache.JG_val, cache.HG_val, cache.k)
end

function compute_matrices(
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    n, k = length(vars), length(G)
    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    ∇G_sys = HC.InterpretedSystem(
        System(reduce(vcat, HC.differentiate(g, vars) for g in G); variables = vars),
    )

    JG = zeros(Float64, k, n)
    HG = zeros(Float64, k * n, n)
    HC.jacobian!(JG, G_sys, point)
    HC.jacobian!(HG, ∇G_sys, point)

    return _compute_matrices(JG, HG, k)
end

# hessian of the routing function on X = V(G) at a point, together with the
# orthonormal tangent basis V it is expressed in. callers that need both (the path
# tracker does) get them from one pass instead of two.
function hessian_and_tangent(cache::RoutingCache, point::AbstractVector{Float64})
    W, V = compute_matrices(cache, point)
    ∇r, Hr = ambient_gradient_hessian!(cache, point)

    H = transpose(V) * Hr * V
    @inbounds for i in eachindex(W)
        H .+= W[i] .* ∇r[i]
    end
    return H, V
end

hessian(cache::RoutingCache, point::AbstractVector{Float64}) =
    first(hessian_and_tangent(cache, point))

hessian(r::RoutingFunction, G::Vector{Expression}, point::Vector{Float64}) =
    hessian(RoutingCache(r, G), point)

# hessian of any Expression f on V(G) at a specified point
function hessian(
    f::Expression,
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    W, V = compute_matrices(G, vars, point)
    ∇f, Hf = ambient_gradient_hessian(f, vars, point)

    H = transpose(V) * Hf * V
    @inbounds for i in eachindex(W)
        H .+= W[i] .* ∇f[i]
    end
    return H
end

# computes the index of a critical point by counting the number of
# LA.eigenvalues of Hr(crit) that match the sign of r(crit)
function idx(
    r::RoutingFunction,
    H::Matrix{Float64},
    critical_point::Vector{Float64}
    )::Int64

    σ = sign(evaluate_r(r, critical_point))
    i = 0
    for eval in LA.eigvals(LA.Symmetric(H))
        if sign(eval) == σ
            i += 1
        end
    end

    return i
end

function sort_routing_points_by_index(
    cache::RoutingCache,
    critical_points::Vector{Vector{Float64}}
    )

    sorter = Dict{Int,Vector{Vector{Float64}}}()

    for P in critical_points
        ind = idx(cache.r, hessian(cache, P), P)
        push!(get!(() -> Vector{Float64}[], sorter, ind), P)
    end

    return sorter
end

sort_routing_points_by_index(
    r::RoutingFunction,
    G::Vector{Expression},
    critical_points::Vector{Vector{Float64}},
) = sort_routing_points_by_index(RoutingCache(r, G), critical_points)
