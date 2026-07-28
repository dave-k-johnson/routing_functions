struct routing_function
    f::Expression               # numerator expression
    g::Expression               # denominator expression
                                    # computed from f and c
    c::Vector{Float64}          # center for numerator
                                    # can be specified, otherwise chosen randomly
    d::Int64                    # degree of num
    eval::Function              # function to HC.evaluate routing function num
    grad::Vector{Expression}    # gradient of routing function
    eval_grad::Function         # HC.evaluates gradient at a point
    vars::Vector{Variable}      # vector of variables.
                                    # if f has all variables present, 
                                    # this will be computed automatically

    function routing_function(f::Expression, vars::Vector{Variable}, c::Vector{Float64})

        g = sum([(vars[i] - c[i])^2 for i in eachindex(vars)]) + 1
        d = degree(f) ÷ 2 + 1
        
        function eval_fun(P::Vector{Float64})
            if length(P) != length(vars)
                error("length of point must match number of variables")
            else
                HC.evaluate(f/g^d, vars => P)
            end
        end

        grad = HC.differentiate(f/g^d, vars)

        function eval_grad(P::Vector{Float64})
            if length(P) != length(vars)
                error("length of point must match number of variables")
            else
                HC.evaluate(grad, vars => P)
            end
        end

        new(f, g, c, d, eval_fun, grad, eval_grad, vars)
    end


end

routing_function(f::Expression, c::Vector{Float64}) = routing_function(f, variables(f), c)

routing_function(f::Expression, vars::Vector{Variable}) = routing_function(f, vars, rand(length(vars)))

routing_function(f::Expression) = routing_function(f, variables(f), rand(length(variables(f))))


function routing_system(
    r::routing_function, 
    G::Vector{Expression}
    )::System
    
    @var λ[1:length(G)]
    ∇f = HC.differentiate(r.f, r.vars)
    ∇g = HC.differentiate(r.g, r.vars)
    ∇r_num = ∇f * r.g - r.d * r.f * ∇g
    eqns = vcat(∇r_num - (r.g)^(r.d + 1) * transpose(HC.differentiate(G, r.vars)) * λ, G)
    return System(eqns; variables=vcat(r.vars, λ))
end

# gradient flow of r restricted to V(G): move along the tangential component of
# sign(r)∇r and correct back onto the variety at every step. the limits are critical
# points of r|V(G), i.e. honest routing points, which is what makes them usable as
# monodromy start solutions.
function flow_to_routing_points(
    r::routing_function,
    G::Vector{Expression},
    sys::System;
    nstarts::Int64 = 100,
    box::Float64 = 3.0,
    tspan::Tuple{Float64,Float64} = (0.0, 200.0),
    reg::Float64 = 1e-8,
    maxiters::Int64 = 10_000,
    f_tol::Float64 = 1e-8
    )::Vector{Vector{ComplexF64}}

    n = length(r.vars)
    JG = HC.differentiate(G, r.vars)
    ∇r_num = HC.differentiate(r.f, r.vars) * r.g - r.d * r.f * HC.differentiate(r.g, r.vars)

    F = HC.InterpretedSystem(sys)
    # `dir` is the ODE parameter: +1 flows to the attracting critical points of
    # r|V(G), -1 to the repelling ones. neither direction alone finds both.
    flow!, f_sys = projected_gradient_field(r, G; reg = reg)

    # r changes sign across V(f), so the field is discontinuous there and the
    # integrator jumps around, so cut it short.
    f_cb = zeros(ComplexF64, 1)
    function on_zero_locus(u, t, integrator)
        HC.evaluate!(f_cb, f_sys, u)
        return abs(real(f_cb[1])) < f_tol
    end
    callback = SciMLBase.DiscreteCallback(on_zero_locus, SciMLBase.terminate!)

    pts = Vector{ComplexF64}[]
    for _ = 1:nstarts, dir in (1.0, -1.0)
        start_pt = box * (2 * rand(n) .- 1)
        project_to_variety!(start_pt, G, r.vars)
        all(isfinite, start_pt) || continue

        prob = SciMLBase.ODEProblem(flow!, start_pt, tspan, dir)
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
        Jx = HC.evaluate(JG, r.vars => x)
        gx = HC.evaluate(r.g, r.vars => x)
        λ = (gx^(r.d + 1) * transpose(Jx)) \ HC.evaluate(∇r_num, r.vars => x)
        newton_result = HC.newton(F, ComplexF64.(vcat(x, λ)))
        HC.is_success(newton_result) || continue
        push!(pts, HC.solution(newton_result))
    end

    return isempty(pts) ? pts : HC.unique_points(pts)
end

function routing_points(
    r::routing_function,
    G::Vector{Expression};
    all_vars::Bool = false,
    zero_tol::Float64 = 1e-5,
    nstarts::Int64 = 100,
    box::Float64 = 3.0,
    verbose::Bool = false
    )::Vector{Vector{Float64}}
    n = length(r.vars)              # ambient dimension, i.e. where ∇r lives
    sys = routing_system(r, G)
    vars = variables(sys)
    N = length(vars)                # n + length(G), the (x, λ) count
    m = length(expressions(sys))

    # the monodromy group of this family is frequently intransitive, so a single
    # random start solution only ever reaches its own orbit. the projected gradient
    # flow supplies real routing points, which sit in exactly the orbits we want.
    seed_points = flow_to_routing_points(r, G, sys; nstarts = nstarts, box = box)

    # subtracting generic affine-linear forms rather than constants enlarges the
    # parameter space enough to make the monodromy group much closer to transitive.
    # the target parameter 0 recovers sys itself.
    @var q[1:m, 1:(N + 1)]
    shifts = [sum(q[i, j] * vars[j] for j = 1:N) + q[i, N+1] for i = 1:m]
    parametrized_system = System(
        [expressions(sys)[i] - shifts[i] for i = 1:m];
        variables = vars,
        parameters = vec(q),
    )

    S0 = randn(ComplexF64, N)
    Q0 = randn(ComplexF64, m, N + 1)
    # choose the constant column so that S0 solves the family over p0
    Q0[:, N+1] .= HC.evaluate(sys, S0) .- Q0[:, 1:N] * S0
    p0 = vec(Q0)
    p_target = zeros(ComplexF64, length(p0))

    # each seed solves the family at p_target, so it has to be tracked onto the
    # generic fibre over p0 individually
    S0_all = Vector{ComplexF64}[S0]
    for pt in seed_points
        res_pt = HC.solve(
            parametrized_system,
            [pt];
            start_parameters = p_target,
            target_parameters = p0,
            show_progress = false,
        )
        append!(S0_all, solutions(res_pt))
    end
    S0_all = HC.unique_points(S0_all)

    mon_result = monodromy_solve(parametrized_system, S0_all, p0; show_progress = false)

    res = HC.solve(
        parametrized_system,
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

    pts = [
        p for p in HC.unique_points(candidates) if
        abs(HC.evaluate(r.f, r.vars => p[1:n])) > zero_tol
    ]

    # sanity check for small systems: res = HC.solve(sys)
    return all_vars ? pts : [p[1:n] for p in pts]
end

