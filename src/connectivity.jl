# reflexive-transitive closure of A: S = ⋁ₖ₌₁ⁿ Aᵏ, in Boolean arithmetic
function boolean_power_sum(A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    n = size(A, 1)

    # Normalize to Bool (accepts 0/1 integers too)
    Ab = Matrix{Bool}(A .!= 0)

    S = falses(n, n)
    P = copy(Ab)      # P = A^k (Boolean)
    Q = falses(n, n)  # scratch for the next power

    for k = 1:n
        @inbounds S .= S .| P
        if k < n
            boolean_matmul!(Q, P, Ab)
            P, Q = Q, P
        end
    end

    return S
end

# Boolean matrix multiplication: (A ⊗ B)[i,j] = any_k (A[i,k] & B[k,j])
function boolean_matmul!(C::AbstractMatrix{Bool}, A::AbstractMatrix{Bool}, B::AbstractMatrix{Bool})
    size(A, 2) == size(B, 1) || throw(ArgumentError("Inner dimensions must match"))
    n, m = size(A)
    _, p = size(B)

    @inbounds for j = 1:p, i = 1:n
        val = false
        for k = 1:m
            if A[i, k] & B[k, j]
                val = true
                break  # short-circuit as soon as we find a witness
            end
        end
        C[i, j] = val
    end
    return C
end

boolean_matmul(A::AbstractMatrix{Bool}, B::AbstractMatrix{Bool}) =
    boolean_matmul!(falses(size(A, 1), size(B, 2)), A, B)

# two routing points are joined when a gradient path leaving a positive-index one
# lands on a given index 0 one; the connected components of V(G) are the connected
# components of the resulting graph.
function find_connectivity_matrix(
    cache::RoutingCache;
    grad_step_size::Float64 = 0.05,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2,
    nstarts::Int64 = 100,
    box::Float64 = 3.0,
    proj_tol::Float64 = 1e-8,
    max_attempts::Int64 = 20,
    verbose::Bool = false
    )

    # the search parameters have to be reachable from here: on a badly scaled
    # system the default box misses the variety entirely
    routPoints = routing_points(
        cache;
        nstarts = nstarts,
        box = box,
        proj_tol = proj_tol,
        max_attempts = max_attempts,
        verbose = verbose,
    )
    index_dict = sort_routing_points_by_index(cache, routPoints)
    final_points = index_dict[0]
    initial_points = vcat([v for (k, v) in index_dict if k != 0]...)

    A = Matrix{Int64}(LA.I, length(routPoints), length(routPoints))

    for P in initial_points
        solns = solve_ivp(cache, P, final_points; grad_step_size = grad_step_size,
                          start_step_size = start_step_size, tol = tol)
        P_ind = nearest_index(P, routPoints)
        for x in solns
            x_ind = nearest_index(x, routPoints)
            A[P_ind, x_ind] = 1
            A[x_ind, P_ind] = 1
        end
    end

    return boolean_power_sum(A), routPoints
end

find_connectivity_matrix(r::RoutingFunction, G::Vector{Expression}; kwargs...) =
    find_connectivity_matrix(RoutingCache(r, G); kwargs...)
