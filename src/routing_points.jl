# the Lagrange system whose real solutions are the critical points of r|V(G).
# g^(d+1)·∇r = ∇r_num is polynomial, so clearing the denominator keeps everything
# in the polynomial world HC.jl wants.
function routing_system(
    r::RoutingFunction,
    G::Vector{Expression}
    )::System

    @var λ[1:length(G)]
    eqns = vcat(
        r.grad_num - (r.g)^(r.d + 1) * transpose(HC.differentiate(G, r.vars)) * λ,
        G,
    )
    return System(eqns; variables = vcat(r.vars, λ))
end

routing_system(cache::RoutingCache)::System = cache.sys

# gradient flow of r restricted to V(G): move along the tangential component of
# sign(r)∇r and correct back onto the variety at every step. the limits are critical
# points of r|V(G), i.e. honest routing points, which is what makes them usable as
# monodromy start solutions.
function flow_to_routing_points(
    cache::RoutingCache;
    nstarts::Int64 = 100,
    box::Float64 = 3.0,
    tspan::Tuple{Float64,Float64} = (0.0, 200.0),
    maxiters::Int64 = 10_000,
    f_tol::Float64 = 1e-8,
    proj_tol::Float64 = 1e-8,
    max_attempts::Int64 = 20,
    verbose::Bool = false
    )::Vector{Vector{ComplexF64}}

    r = cache.r
    n = cache.n

    # r changes sign across V(f), so the field is discontinuous there and the
    # integrator jumps around, so cut it short.
    f_cb = zeros(Float64, 1)
    function on_zero_locus(u, t, integrator)
        HC.evaluate!(f_cb, r.f_sys, u)
        return abs(@inbounds f_cb[1]) < f_tol
    end
    callback = SciMLBase.DiscreteCallback(on_zero_locus, SciMLBase.terminate!)

    pts = Vector{ComplexF64}[]
    start_pt = zeros(Float64, n)
    started = 0
    attempts = 0
    attempt_budget = max_attempts * nstarts

    # `dir` is the ODE parameter: +1 flows to the attracting critical points of
    # r|V(G), -1 to the repelling ones. neither direction alone finds both.
    while started < nstarts && attempts < attempt_budget
        attempts += 1

        @inbounds for i = 1:n
            start_pt[i] = box * (2 * rand() - 1)
        end

        # a start that never reached V(G) is not worth integrating: the field's
        # normal term has to drag it back before the tangential part means
        # anything, which is most of the cost and none of the answer. resample
        # instead, so `nstarts` still counts flows rather than attempts.
        _, residual = project_to_variety_residual!(start_pt, cache)
        (isfinite(residual) && residual < proj_tol && all(isfinite, start_pt)) || continue
        started += 1

        for dir in (1.0, -1.0)
            prob = SciMLBase.ODEProblem(cache.flow!, copy(start_pt), tspan, dir)
            sol = SciMLBase.solve(
                prob,
                reltol = 1e-8,
                abstol = 1e-8,
                maxiters = maxiters,
                callback = callback,
            )
            x = last(sol.u)
            all(isfinite, x) || continue

            # recover λ by least squares, then Newton on the full routing system
            jacobian_G!(cache, x)
            HC.evaluate!(cache.grad_num_val, r.grad_num_sys, x)
            gx = evaluate_g(r, x)
            λ = (gx^(r.d + 1) * transpose(cache.JG_val)) \ cache.grad_num_val
            newton_result = HC.newton(cache.sys_interp, ComplexF64.(vcat(x, λ)))
            HC.is_success(newton_result) || continue
            push!(pts, HC.solution(newton_result))
        end
    end

    if verbose || (started < nstarts)
        msg = "flow_to_routing_points: $started of $nstarts starts landed on V(G) " *
              "in $attempts attempts (box = $box)"
        started < nstarts ? @warn(msg * "; raise `box` or `max_attempts`") : println(msg)
    end

    return isempty(pts) ? pts : HC.unique_points(pts)
end

function routing_points(
    cache::RoutingCache;
    all_vars::Bool = false,
    zero_tol::Float64 = 1e-5,
    nstarts::Int64 = 100,
    box::Float64 = 3.0,
    proj_tol::Float64 = 1e-8,
    max_attempts::Int64 = 20,
    verbose::Bool = false
    )::Vector{Vector{Float64}}

    r = cache.r
    n = cache.n                     # ambient dimension, i.e. where ∇r lives
    N = cache.N                     # n + length(G), the (x, λ) count
    m = cache.m

    # the monodromy group of this family is frequently intransitive, so a single
    # random start solution only ever reaches its own orbit. the projected gradient
    # flow supplies real routing points, which sit in exactly the orbits we want.
    seed_points = flow_to_routing_points(
        cache;
        nstarts = nstarts,
        box = box,
        proj_tol = proj_tol,
        max_attempts = max_attempts,
        verbose = verbose,
    )

    S0 = randn(ComplexF64, N)
    Q0 = randn(ComplexF64, m, N + 1)
    # choose the constant column so that S0 solves the family over p0
    Q0[:, N+1] .= HC.evaluate(cache.sys, S0) .- Q0[:, 1:N] * S0
    p0 = vec(Q0)
    p_target = zeros(ComplexF64, length(p0))

    # each seed solves the family at p_target, so it has to be tracked onto the
    # generic fibre over p0 individually
    S0_all = Vector{ComplexF64}[S0]
    for pt in seed_points
        res_pt = HC.solve(
            cache.param_sys,
            [pt];
            start_parameters = p_target,
            target_parameters = p0,
            show_progress = false,
        )
        append!(S0_all, solutions(res_pt))
    end
    S0_all = HC.unique_points(S0_all)

    mon_result = monodromy_solve(cache.param_sys, S0_all, p0; show_progress = false)

    res = HC.solve(
        cache.param_sys,
        solutions(mon_result);
        start_parameters = p0,
        target_parameters = p_target,
        show_progress = false,
    )

    if verbose
        println(
            "routing_points: $(length(seed_points)) flow seeds, $(length(S0_all)) start solutions, ",
            "$(length(solutions(mon_result))) points on the generic fibre ($(mon_result.returncode))",
        )
    end

    # the flow seeds are Newton-certified solutions of sys, so keep them even when
    # the homotopy back to p_target loses the corresponding path
    candidates = vcat(
        HC.real_solutions(res),
        [real.(p) for p in seed_points if maximum(abs.(imag.(p))) < 1e-8],
    )
    isempty(candidates) && return Vector{Float64}[]

    pts = [p for p in HC.unique_points(candidates) if
           abs(evaluate_r(r, view(p, 1:n))) > zero_tol]

    # sanity check for small systems: res = HC.solve(cache.sys)
    return all_vars ? pts : [p[1:n] for p in pts]
end

# convenience wrappers: build a cache on the fly. call these only once -- if you
# are going to touch the same (r, G) more than once, build the cache yourself and
# pass it around.
flow_to_routing_points(r::RoutingFunction, G::Vector{Expression}; kwargs...) =
    flow_to_routing_points(RoutingCache(r, G); kwargs...)

routing_points(r::RoutingFunction, G::Vector{Expression}; kwargs...) =
    routing_points(RoutingCache(r, G); kwargs...)
