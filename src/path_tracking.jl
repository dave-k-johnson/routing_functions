# Newton correction onto V(G), along the normal directions. returns the point
# together with the residual ‖G(point)‖ actually achieved, so callers can tell a
# point that reached the variety from one that merely ran out of iterations.
#
# the pseudo-inverse is regularised so that a rank-deficient Jacobian (a singular
# point of V(G)) damps the step instead of blowing it up.
#
# the undamped step is not reliable away from the variety -- on 3RPR it routinely
# overshoots and sends the residual up by three orders of magnitude -- so each step
# is backtracked until it actually decreases ‖G‖. that makes the iteration
# monotone, which in turn makes "the residual stopped improving" a meaningful
# stopping rule: `tol` is absolute and on a badly scaled system sits below
# anything the residual can reach.
function _project_to_variety!(
    point::AbstractVector{Float64},
    G_sys::HC.InterpretedSystem,
    G_val::Vector{Float64},
    JG_val::Matrix{Float64},
    M::Matrix{Float64},
    wk::Vector{Float64},
    step::Vector{Float64},
    base::Vector{Float64},
    maxiter::Int64,
    tol::Float64,
    reg::Float64,
    maxbacktrack::Int64
    )

    residual = Inf

    for _ = 1:maxiter
        HC.evaluate_and_jacobian!(G_val, JG_val, G_sys, point)
        residual = LA.norm(G_val)
        residual < tol && return point, residual

        # step = Jᵀ (J Jᵀ + reg·I)⁻¹ G
        copyto!(wk, G_val)
        normal_factor!(M, JG_val, reg)
        normal_solve!(wk, M)
        LA.mul!(step, transpose(JG_val), wk)

        copyto!(base, point)
        α = 1.0
        accepted = false
        for _ = 1:maxbacktrack
            @inbounds for i in eachindex(point)
                point[i] = base[i] - α * step[i]
            end
            HC.evaluate!(G_val, G_sys, point)
            if LA.norm(G_val) < residual
                accepted = true
                break
            end
            α *= 0.5
        end

        # no step length improves the residual: this is as close as we get
        if !accepted
            copyto!(point, base)
            return point, residual
        end
    end

    return point, residual
end

# `maxiter` is generous because the loop exits on its own once the residual stops
# improving; a point that projects cleanly still costs only a few iterations.
project_to_variety!(
    point::AbstractVector{Float64},
    cache::RoutingCache;
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = 1e-8,
    maxbacktrack::Int64 = 20,
) = first(_project_to_variety!(point, cache.G_sys, cache.G_val, cache.JG_val,
                               cache.M, cache.wk, cache.wn, cache.wn2,
                               maxiter, tol, reg, maxbacktrack))

function project_to_variety!(
    point::AbstractVector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable};
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = 1e-8,
    maxbacktrack::Int64 = 20
    )

    n, k = length(vars), length(G)
    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    return first(_project_to_variety!(point, G_sys, zeros(k), zeros(k, n), zeros(k, k),
                                      zeros(k), zeros(n), zeros(n),
                                      maxiter, tol, reg, maxbacktrack))
end

# same, but reporting the residual reached. `flow_to_routing_points` uses this to
# throw away starts that never made it onto the variety.
project_to_variety_residual!(
    point::AbstractVector{Float64},
    cache::RoutingCache;
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = 1e-8,
    maxbacktrack::Int64 = 20,
) = _project_to_variety!(point, cache.G_sys, cache.G_val, cache.JG_val, cache.M,
                         cache.wk, cache.wn, cache.wn2, maxiter, tol, reg, maxbacktrack)

# minimum distance from `point` to an index 0 routing point. this runs inside the
# ODE callback, so it stays allocation free.
function distance_to_endpoints(
    point::AbstractVector{Float64},
    index0_points::Vector{Vector{Float64}}
    )::Float64

    best = Inf
    @inbounds for Q in index0_points
        s = 0.0
        for i in eachindex(point, Q)
            δ = Q[i] - point[i]
            s += δ * δ
        end
        s < best && (best = s)
    end
    return sqrt(best)
end

# index of the point of `points` nearest to `point`
function nearest_index(
    point::AbstractVector{Float64},
    points::Vector{Vector{Float64}}
    )::Int64

    best = Inf
    ind = 0
    @inbounds for (j, Q) in enumerate(points)
        s = 0.0
        for i in eachindex(point, Q)
            δ = Q[i] - point[i]
            s += δ * δ
        end
        if s < best
            best = s
            ind = j
        end
    end
    return ind
end

# projected gradient field of r on X = V(G): the tangential component of
# sign(r)∇r, plus a Newton correction that holds the path on X. this is the field
# both the routing point search and the path tracker integrate.
#
# `dir` is the ODE parameter: +1 follows the flow, -1 reverses it.
# `unit_speed` rescales the tangential part so that time is arc length. the path
# tracker needs this -- it starts next to a critical point, where ∇r ≈ 0 and an
# unscaled field would crawl -- while the routing point search does not.
#
# the closure owns its buffers rather than borrowing the cache's, so that an
# integration in progress cannot be disturbed by anything else touching the cache.
function _projected_gradient_field(
    r::RoutingFunction,
    G_sys::HC.InterpretedSystem,
    n::Int64,
    k::Int64,
    reg::Float64,
    unit_speed::Bool
    )

    grad_sys = r.grad_sys
    f_sys = r.f_sys

    G_val = zeros(Float64, k)
    JG_val = zeros(Float64, k, n)
    grad_val = zeros(Float64, n)
    f_val = zeros(Float64, 1)
    M = zeros(Float64, k, k)
    wk = zeros(Float64, k)
    wk2 = zeros(Float64, k)
    tangential = zeros(Float64, n)

    function flow!(du, u, dir, t)
        HC.evaluate_and_jacobian!(G_val, JG_val, G_sys, u)
        HC.evaluate!(grad_val, grad_sys, u)
        HC.evaluate!(f_val, f_sys, u)

        # both halves of the field solve against J Jᵀ + reg·I, so factor it once
        normal_factor!(M, JG_val, reg)

        # tangential part: (I - Jᵀ(JJᵀ + reg·I)⁻¹J) ∇r, oriented by sign(r).
        # g > 0 on ℝⁿ, so sign(r) == sign(f).
        LA.mul!(wk, JG_val, grad_val)
        normal_solve!(wk, M)
        LA.mul!(tangential, transpose(JG_val), wk)
        s = dir * sign(@inbounds f_val[1])
        @inbounds for i = 1:n
            tangential[i] = s * (grad_val[i] - tangential[i])
        end

        if unit_speed
            speed = LA.norm(tangential)
            speed > reg && (tangential ./= speed)
        end

        # normal part: pull back onto V(G)
        copyto!(wk2, G_val)
        normal_solve!(wk2, M)
        LA.mul!(du, transpose(JG_val), wk2)
        @inbounds for i = 1:n
            du[i] = tangential[i] - du[i]
        end
        return nothing
    end

    return flow!
end

# the fields are built once, when the cache is. `reg` is fixed at that point --
# pass it to `RoutingCache` if you need something other than the default.
#
# returns the in-place field together with the interpreted system for r's
# numerator, which callers use to build callbacks on the zero locus of r.
projected_gradient_field(cache::RoutingCache; unit_speed::Bool = false) =
    (unit_speed ? cache.flow_unit! : cache.flow!), cache.r.f_sys

projected_gradient_field(
    r::RoutingFunction,
    G::Vector{Expression};
    reg::Float64 = 1e-8,
    unit_speed::Bool = false,
) = projected_gradient_field(RoutingCache(r, G; reg = reg); unit_speed = unit_speed)

# flows from `point` along the projected gradient until it reaches one of
# `index0_points`, overwriting `point` with where it landed and returning whether
# it got there. the solver picks the step size adaptively, and a root-finding
# callback locates the arrival instead of overshooting it by a fixed step.
function gradient_flow!(
    cache::RoutingCache,
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}};
    tol::Float64 = 1e-2,
    dtmax::Float64 = 0.0,
    tspan::Tuple{Float64,Float64} = (0.0, 1e3),
    maxiters::Int64 = 10^6,
    Verbose::Bool = false
    )::Bool

    # the callback fires on a sign change, so a path that starts inside the ball
    # would never trigger it
    if distance_to_endpoints(point, index0_points) <= tol
        Verbose && println("gradient_flow!: start point is already within tol of an endpoint")
        return true
    end

    # unit speed makes t arc length, so this fires the moment the path first comes
    # within tol of an index 0 routing point
    arrived(u, t, integrator) = distance_to_endpoints(u, index0_points) - tol
    callback = SciMLBase.ContinuousCallback(arrived, SciMLBase.terminate!)

    # capping the step at tol keeps a single step from jumping clean over the ball,
    # which the root finder would then never see
    prob = SciMLBase.ODEProblem(cache.flow_unit!, copy(point), tspan, 1.0)
    sol = SciMLBase.solve(
        prob,
        reltol = 1e-8,
        abstol = 1e-8,
        dtmax = dtmax > 0 ? dtmax : tol,
        maxiters = maxiters,
        callback = callback,
    )

    point .= last(sol.u)
    arrived_at_endpoint = sol.retcode == SciMLBase.ReturnCode.Terminated

    if Verbose
        println(
            "gradient_flow!: ", sol.retcode, ", arc length ", round(last(sol.t); digits = 4),
            ", distance to nearest index 0 point ",
            round(distance_to_endpoints(point, index0_points); digits = 6),
        )
    end

    return arrived_at_endpoint
end

gradient_flow!(
    r::RoutingFunction,
    G::Vector{Expression},
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}};
    reg::Float64 = 1e-8,
    kwargs...,
) = gradient_flow!(RoutingCache(r, G; reg = reg), point, index0_points; kwargs...)

# finds P ± ϵv for unstable LA.eigenvectors v at a critical point P
function find_starting_points_for_flow(
    cache::RoutingCache,
    critical_point::Vector{Float64},
    H::Matrix{Float64}, # hessian of r at crit_point
    V::Matrix{Float64}; # basis of tangent space of V(G) at critical_point
    step_size::Float64 = 0.1
    )

    d = size(V, 2)

    sgn = sign(evaluate_r(cache.r, critical_point))
    E = LA.eigen(LA.Symmetric(H))
    unstable_vecs = [V * view(E.vectors, :, i) for i = 1:d if sign(E.values[i]) == sgn]

    starts = Vector{Float64}[]
    for v in unstable_vecs
        push!(starts, project_to_variety!(critical_point + v * step_size, cache))
        push!(starts, project_to_variety!(critical_point - v * step_size, cache))
    end
    return starts
end

# leaves `initial_point` along each unstable direction and follows the flow to the
# index 0 routing points it reaches. paths that never arrive are dropped rather
# than reported at wherever they stalled.
function solve_ivp(
    cache::RoutingCache,
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    grad_step_size::Float64 = 0.0,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2,
    Verbose::Bool = false
    )::Vector{Vector{Float64}}

    # one pass gives both the hessian on V(G) and the tangent basis it lives in
    H, V = hessian_and_tangent(cache, initial_point)
    starts = find_starting_points_for_flow(cache, initial_point, H, V; step_size = start_step_size)

    solns = Vector{Float64}[]
    for P in starts
        Q = copy(P)
        if gradient_flow!(cache, Q, final_points; tol = tol, dtmax = grad_step_size,
                          Verbose = Verbose)
            push!(solns, Q)
        elseif Verbose
            println("solve_ivp: flow from ", round.(P; digits = 6), " reached no index 0 routing point")
        end
    end

    return solns
end

solve_ivp(
    r::RoutingFunction,
    G::Vector{Expression},
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    kwargs...,
) = solve_ivp(RoutingCache(r, G), initial_point, final_points; kwargs...)
