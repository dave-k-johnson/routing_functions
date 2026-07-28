# everything that can be precomputed from the pair (r, V(G)) and then reused:
# the symbolic derivatives of G compiled into interpreted systems, the routing
# system and the parametrised family monodromy runs over, the two projected
# gradient fields, and the scratch buffers all of those write into.
#
# building a cache is the expensive part -- every symbolic `differentiate` and
# every `InterpretedSystem` construction happens once, here -- after which the
# numerical routines only touch preallocated arrays.
#
# NOTE: the buffers make a cache stateful. one cache must not be used from two
# threads at once; build one per thread instead.
struct RoutingCache{F,Fu}
    r::RoutingFunction
    G::Vector{Expression}
    vars::Vector{Variable}
    n::Int64                        # ambient dimension
    k::Int64                        # number of defining equations of V(G)
    reg::Float64                    # pseudo-inverse regularisation of the flow fields

    G_sys::HC.InterpretedSystem     # G; its jacobian is JG
    ∇G_sys::HC.InterpretedSystem    # [∇g₁; …; ∇gₖ]; its jacobian stacks the hessians of the gᵢ

    sys::System                     # routing system in (x, λ)
    sys_interp::HC.InterpretedSystem
    sys_vars::Vector{Variable}      # vcat(vars, λ)
    param_sys::System               # sys minus generic affine-linear forms
    m::Int64                        # number of equations of sys
    N::Int64                        # n + k, the (x, λ) count

    flow!::F                        # projected gradient field of r on V(G)
    flow_unit!::Fu                  # same field, rescaled so that time is arc length

    # scratch, sized once. the flow fields keep their own private copies so that
    # an integration in progress cannot be clobbered by a call to, say, `hessian`.
    G_val::Vector{Float64}          # k
    JG_val::Matrix{Float64}         # k × n
    HG_val::Matrix{Float64}         # (k·n) × n; rows (i-1)n+1:in are the hessian of gᵢ
    grad_val::Vector{Float64}       # n
    Hr_val::Matrix{Float64}         # n × n
    grad_num_val::Vector{Float64}   # n
    M::Matrix{Float64}              # k × k
    wk::Vector{Float64}             # k
    wn::Vector{Float64}             # n
    wn2::Vector{Float64}            # n
end

function RoutingCache(
    r::RoutingFunction,
    G::Vector{Expression};
    reg::Float64 = 1e-8,
    )

    vars = r.vars
    n = length(vars)
    k = length(G)

    # G has to live in r's variables. HC's own complaint here is
    # "Not all variables or parameters of the system are given", which names the
    # variables but not the mismatch that caused it -- and the usual cause is a
    # stale r left over from another problem in the same session.
    stray = setdiff(variables(G), vars)
    isempty(stray) || error(
        "G involves variables that r does not: ", join(stray, ", "),
        ". r is a routing function in ", join(vars, ", "),
        " -- rebuild it over the variables of G.",
    )

    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    # stacked row-wise so that the jacobian's i-th n × n block is exactly ∇²gᵢ
    ∇G_sys = HC.InterpretedSystem(
        System(reduce(vcat, HC.differentiate(g, vars) for g in G); variables = vars),
    )

    sys = routing_system(r, G)
    sys_vars = variables(sys)
    m = length(expressions(sys))
    N = length(sys_vars)

    # subtracting generic affine-linear forms rather than constants enlarges the
    # parameter space enough to make the monodromy group much closer to transitive.
    # the target parameter 0 recovers sys itself.
    @var q[1:m, 1:(N + 1)]
    shifts = [sum(q[i, j] * sys_vars[j] for j = 1:N) + q[i, N+1] for i = 1:m]
    param_sys = System(
        [expressions(sys)[i] - shifts[i] for i = 1:m];
        variables = sys_vars,
        parameters = vec(q),
    )

    flow! = _projected_gradient_field(r, G_sys, n, k, reg, false)
    flow_unit! = _projected_gradient_field(r, G_sys, n, k, reg, true)

    return RoutingCache(
        r, G, vars, n, k, reg,
        G_sys, ∇G_sys,
        sys, HC.InterpretedSystem(sys), sys_vars, param_sys, m, N,
        flow!, flow_unit!,
        zeros(Float64, k),          # G_val
        zeros(Float64, k, n),       # JG_val
        zeros(Float64, k * n, n),   # HG_val
        zeros(Float64, n),          # grad_val
        zeros(Float64, n, n),       # Hr_val
        zeros(Float64, n),          # grad_num_val
        zeros(Float64, k, k),       # M
        zeros(Float64, k),          # wk
        zeros(Float64, n),          # wn
        zeros(Float64, n),          # wn2
    )
end

RoutingCache(r::RoutingFunction, g::Expression; kwargs...) =
    RoutingCache(r, [g]; kwargs...)

function Base.show(io::IO, cache::RoutingCache)
    print(io, "RoutingCache: ", cache.r, " on V(G) ⊂ ℝ^", cache.n,
          " cut out by ", cache.k, " equation", cache.k == 1 ? "" : "s")
end

# G(x) and JG(x), written into the cache buffers
function evaluate_G!(cache::RoutingCache, point::AbstractVector{Float64})
    HC.evaluate_and_jacobian!(cache.G_val, cache.JG_val, cache.G_sys, point)
    return cache.G_val, cache.JG_val
end

# JG(x) alone
function jacobian_G!(cache::RoutingCache, point::AbstractVector{Float64})
    HC.jacobian!(cache.JG_val, cache.G_sys, point)
    return cache.JG_val
end

# M ← the Cholesky factor of J Jᵀ + reg·I, kept in M's lower triangle. the
# regularisation keeps a rank-deficient jacobian (a singular point of V(G)) from
# blowing the step up instead of merely damping it, and makes the matrix SPD so
# that Cholesky always succeeds.
#
# k is the codimension, so this is typically 1 × 1 or 2 × 2 and LAPACK's dispatch
# and error-checking overhead swamps the arithmetic; hence the hand-rolled version.
function normal_factor!(M::AbstractMatrix{Float64}, J::AbstractMatrix{Float64}, reg::Float64)
    LA.mul!(M, J, transpose(J))
    k = size(M, 1)
    @inbounds for j = 1:k
        s = M[j, j] + reg
        for p = 1:j-1
            s -= M[j, p]^2
        end
        Ljj = sqrt(s)
        M[j, j] = Ljj
        for i = j+1:k
            s = M[i, j]
            for p = 1:j-1
                s -= M[i, p] * M[j, p]
            end
            M[i, j] = s / Ljj
        end
    end
    return M
end

# solves L Lᵀ y = y in place, L being the factor left by `normal_factor!`
function normal_solve!(y::AbstractVector{Float64}, L::AbstractMatrix{Float64})
    k = length(y)
    @inbounds for i = 1:k
        s = y[i]
        for p = 1:i-1
            s -= L[i, p] * y[p]
        end
        y[i] = s / L[i, i]
    end
    @inbounds for i = k:-1:1
        s = y[i]
        for p = i+1:k
            s -= L[p, i] * y[p]
        end
        y[i] = s / L[i, i]
    end
    return y
end
