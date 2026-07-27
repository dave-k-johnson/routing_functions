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

function routing_points(
    r::routing_function, 
    G::Vector{Expression};
    all_vars::Bool = false,
    zero_tol::Float64 = 1e-5
    )::Vector{Vector{Float64}}
    n = length(r.vars)              # ambient dimension, i.e. where ∇r lives
    sys = routing_system(r, G)
    m = length(expressions(sys))    # n + length(G), the (x, λ) count

    #find start fibre using monodromy
    S0 = rand(ComplexF64, length(variables(sys)))
    rhs0 = HC.evaluate(sys,S0)

    @var dummy[1:m]
    parametrized_eqs = [expressions(sys)[i] - dummy[i] for i in 1:m]
    parametrized_system = HC.System(parametrized_eqs; variables=variables(sys), parameters = dummy)

    #expand start solutions using gradient flow on ∇r
    ∇r = HC.InterpretedSystem(HC.System(r.grad; variables = r.vars))
    ∇r_val = zeros(ComplexF64, n)
    # in-place so that du stays a Vector{Float64}; the interpreter tape is complex
    function flow!(du, u, param, t)
        HC.evaluate!(∇r_val, ∇r, u)
        du .= real.(∇r_val)
        return nothing
    end

    new_pts = Vector{ComplexF64}[]
    # Setting up grid
    start_grid_center = zeros(n)
    start_grid_width = 5
    start_grid_stepsize = 0.2
    w = (start_grid_width / 2)
    grid = [
        (start_grid_center[i]-w):start_grid_stepsize:(start_grid_center[i]+w) for
        i = 1:n
    ]

    tspan = (0.0, 1e4)
    for start_point in Iterators.product(grid...)

            start_pt = collect(Float64, start_point)
            prob = SciMLBase.ODEProblem(flow!, start_pt, tspan)
            sol = SciMLBase.solve(prob, reltol=1e-6, abstol=1e-6)
            convergence_point = ComplexF64.(last(sol.u))
            newton_result = HC.newton(∇r, convergence_point)
            HC.is_success(newton_result) || continue
            # ∇r vanishes here, so λ = 0 satisfies the first block of sys exactly
            push!(new_pts, vcat(HC.solution(newton_result), zeros(ComplexF64, m - n)))

    end
    if length(new_pts) > 0
        new_pts = HC.unique_points(new_pts)
    end

    # each flow point solves parametrized_system at its own parameter value,
    # so it has to be tracked onto the fibre over rhs0 individually
    S0_new_sols = Vector{ComplexF64}[]
    for pt in new_pts
        res_pt = HC.solve(
            parametrized_system,
            [pt];
            start_parameters = HC.evaluate(sys, pt),
            target_parameters = rhs0,
            show_progress = false,
        )
        append!(S0_new_sols, solutions(res_pt))
    end

    S0 = HC.unique_points([[S0]; S0_new_sols])
    mon_result = monodromy_solve(parametrized_system, S0, rhs0)

    intermediate_rhs = randn(ComplexF64, length(rhs0))
    result_intermediate = HC.solve(parametrized_system, solutions(mon_result); start_parameters = rhs0, target_parameters = intermediate_rhs)



    res = HC.solve(parametrized_system, result_intermediate; start_parameters = intermediate_rhs, target_parameters = zeros(ComplexF64, length(rhs0)))

    # res = HC.solve(sys)

    if all_vars
        return [p for p in HC.real_solutions(res) if abs(HC.evaluate(r.f, r.vars => p)) > zero_tol]
    else
        return [p[1:length(r.vars)] for p in HC.real_solutions(res) if abs(HC.evaluate(r.f, r.vars => p[1:length(r.vars)])) > zero_tol]
    end
end

