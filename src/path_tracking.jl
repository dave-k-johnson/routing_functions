# Newton correction onto V(G), along the normal directions.
# the pseudo-inverse is regularised so that a rank-deficient Jacobian (a singular
# point of V(G)) damps the step instead of blowing it up.
function project_to_variety!(
    point::Vector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable};
    maxiter::Int64 = 5,
    tol::Float64 = 1e-15,
    reg::Float64 = 1e-8
    )

    JG = HC.differentiate(G, vars)

    for _ in 1:maxiter
        Gx = HC.evaluate(G, vars => point)
        LA.norm(Gx) < tol && return point

        JGx = HC.evaluate(JG, vars => point)
        point .-= transpose(JGx) * ((JGx * transpose(JGx) + reg * LA.I) \ Gx)
    end

    return point
end

# computes minimum distance to an index 0 saddle point
function distance_to_endpoints(
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}}
    )

    return minimum(LA.norm.([Q - point for Q in index0_points]))

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
# returns the in-place field together with the interpreted system for r's
# numerator, which callers use to build callbacks on the zero locus of r.
function projected_gradient_field(
    r::routing_function,
    G::Vector{Expression};
    reg::Float64 = 1e-8,
    unit_speed::Bool = false
    )

    n = length(r.vars)
    k = length(G)
    JG = HC.differentiate(G, r.vars)

    ∇r_sys = HC.InterpretedSystem(System(r.grad; variables = r.vars))
    JG_sys = HC.InterpretedSystem(System(vec(JG); variables = r.vars))
    G_sys = HC.InterpretedSystem(System(G; variables = r.vars))
    f_sys = HC.InterpretedSystem(System([r.f]; variables = r.vars))

    # in-place so that du stays a Vector{Float64}; the interpreter tape is complex
    ∇r_val = zeros(ComplexF64, n)
    JG_val = zeros(ComplexF64, k * n)
    G_val = zeros(ComplexF64, k)
    f_val = zeros(ComplexF64, 1)
    tangential = zeros(Float64, n)

    function flow!(du, u, dir, t)
        HC.evaluate!(∇r_val, ∇r_sys, u)
        HC.evaluate!(JG_val, JG_sys, u)
        HC.evaluate!(G_val, G_sys, u)
        HC.evaluate!(f_val, f_sys, u)

        J = reshape(real.(JG_val), k, n)
        grad = real.(∇r_val)
        # regularised pseudo-inverse: at a singular point of X the Jacobian drops
        # rank and an exact J \ G blows up, which stalls the integrator.
        M = J * transpose(J) + reg * LA.I
        # g > 0 on ℝⁿ, so sign(r) == sign(f)
        tangential .= dir * sign(real(f_val[1])) * (grad - transpose(J) * (M \ (J * grad)))

        if unit_speed
            speed = LA.norm(tangential)
            speed > reg && (tangential ./= speed)
        end

        du .= tangential - transpose(J) * (M \ real.(G_val))
        return nothing
    end

    return flow!, f_sys
end

# flows from `point` along the projected gradient until it reaches one of
# `index0_points`, overwriting `point` with where it landed and returning whether
# it got there. the solver picks the step size adaptively, and a root-finding
# callback locates the arrival instead of overshooting it by a fixed step.
function gradient_flow!(
    r::routing_function,
    G::Vector{Expression},
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}};
    tol::Float64 = 1e-2,
    dtmax::Float64 = 0.0,
    tspan::Tuple{Float64,Float64} = (0.0, 1e3),
    maxiters::Int64 = 10^6,
    reg::Float64 = 1e-8,
    field = nothing,
    Verbose::Bool = false
    )::Bool

    # the callback fires on a sign change, so a path that starts inside the ball
    # would never trigger it
    if distance_to_endpoints(point, index0_points) <= tol
        Verbose && println("gradient_flow!: start point is already within tol of an endpoint")
        return true
    end

    # a caller tracking several paths can build the field once and hand it in; it
    # carries mutable scratch buffers, so it must not be shared across threads
    flow! = isnothing(field) ? first(projected_gradient_field(r, G; reg = reg, unit_speed = true)) : field

    # unit speed makes t arc length, so this fires the moment the path first comes
    # within tol of an index 0 routing point
    arrived(u, t, integrator) = distance_to_endpoints(u, index0_points) - tol
    callback = SciMLBase.ContinuousCallback(arrived, SciMLBase.terminate!)

    # capping the step at tol keeps a single step from jumping clean over the ball,
    # which the root finder would then never see
    prob = SciMLBase.ODEProblem(flow!, copy(point), tspan, 1.0)
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

# finds P ± ϵv for unstable LA.eigenvectors v at a critical point P
function find_starting_points_for_flow(
    critical_point::Vector{Float64},
    H::Matrix{Float64}, # hessian of r at crit_point
    V::Matrix{Float64},  # basis of tangent space of V(G) at V
    G::Vector{Expression},
    r::routing_function;
    step_size::Float64 = 0.1
    )

    n, d = size(V)

    sgn = sign(r.eval(critical_point))
    E = LA.eigen(H)
    unstable_vecs = [
        V * E.vectors[:,i]
        for i in 1:d
            if sign(E.values[i]) == sgn
    ]

    pos_starting_points = [project_to_variety!(critical_point + v*step_size, G, r.vars) for v in unstable_vecs]
    neg_starting_points = [project_to_variety!(critical_point - v*step_size, G, r.vars) for v in unstable_vecs]

    return vcat(pos_starting_points, neg_starting_points)
end


# leaves `initial_point` along each unstable direction and follows the flow to the
# index 0 routing points it reaches. paths that never arrive are dropped rather
# than reported at wherever they stalled.
function solve_ivp(
    r::routing_function,
    G::Vector{Expression},
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    grad_step_size::Float64 = 0.0,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2,
    Verbose::Bool = false
    )::Vector{Vector{Float64}}

    V = LA.nullspace(HC.evaluate(HC.differentiate(G, r.vars), r.vars => initial_point))
    H = hessian(r, G, initial_point)
    starts = find_starting_points_for_flow(initial_point, H, V, G, r; step_size = start_step_size)

    solns = Vector{Float64}[]
    # built once and reused by every path out of this critical point
    flow!, _ = projected_gradient_field(r, G; unit_speed = true)

    for P in starts
        Q = copy(P)
        if gradient_flow!(r, G, Q, final_points; tol = tol, dtmax = grad_step_size,
                          field = flow!, Verbose = Verbose)
            push!(solns, Q)
        elseif Verbose
            println("solve_ivp: flow from ", round.(P; digits = 6), " reached no index 0 routing point")
        end
    end

    return solns
end
