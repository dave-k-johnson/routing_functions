# r = f / g^d, where g = ∑ᵢ (xᵢ - cᵢ)² + 1 is strictly positive on ℝⁿ and
# d = deg(f) ÷ 2 + 1 is large enough that r → 0 at infinity.
#
# the struct carries the symbolic data together with the interpreted systems built
# from it. those systems are compiled once here instead of on every evaluation --
# `HC.evaluate(expr, vars => P)` rebuilds an interpreter tape each call, which is
# what made the old closure fields so expensive.
#
# anything that also depends on the variety V(G) lives in `RoutingCache`.
struct RoutingFunction
    f::Expression                # numerator
    g::Expression                # base of the denominator, determined by c
    c::Vector{Float64}           # center of g; chosen randomly unless specified
    d::Int64                     # exponent of g
    vars::Vector{Variable}       # ambient variables
    grad::Vector{Expression}     # ∇r
    grad_num::Vector{Expression} # ∇f·g - d·f·∇g, i.e. g^(d+1)·∇r

    r_sys::HC.InterpretedSystem        # [r]
    f_sys::HC.InterpretedSystem        # [f]; g > 0, so sign(r) == sign(f)
    grad_sys::HC.InterpretedSystem     # ∇r; its jacobian is the ambient hessian of r
    grad_num_sys::HC.InterpretedSystem # ∇f·g - d·f·∇g

    function RoutingFunction(f::Expression, vars::Vector{Variable}, c::Vector{Float64})
        length(c) == length(vars) ||
            error("length of center must match number of variables")

        # f may use fewer variables than `vars` -- the ambient space is whatever
        # the caller says it is -- but it must not use more
        stray = setdiff(variables(f), vars)
        isempty(stray) || error(
            "f involves variables not in vars: ", join(stray, ", "),
            ". vars = ", join(vars, ", "),
        )

        g = sum((vars[i] - c[i])^2 for i in eachindex(vars)) + 1
        d = degree(f) ÷ 2 + 1
        r = f / g^d

        grad = HC.differentiate(r, vars)
        grad_num = HC.differentiate(f, vars) * g - d * f * HC.differentiate(g, vars)

        new(
            f, g, c, d, vars, grad, grad_num,
            HC.InterpretedSystem(System([r]; variables = vars)),
            HC.InterpretedSystem(System([f]; variables = vars)),
            HC.InterpretedSystem(System(grad; variables = vars)),
            HC.InterpretedSystem(System(grad_num; variables = vars)),
        )
    end
end

RoutingFunction(f::Expression, c::Vector{Float64}) = RoutingFunction(f, variables(f), c)

RoutingFunction(f::Expression, vars::Vector{Variable}) =
    RoutingFunction(f, vars, rand(length(vars)))

RoutingFunction(f::Expression) = RoutingFunction(f, variables(f))

Base.length(r::RoutingFunction) = length(r.vars)

function Base.show(io::IO, r::RoutingFunction)
    print(io, "RoutingFunction in ", length(r.vars), " variables: (", r.f, ") / (", r.g, ")^", r.d)
end

# r(P)
function evaluate_r(r::RoutingFunction, point::AbstractVector{<:Real})::Float64
    length(point) == length(r.vars) ||
        error("length of point must match number of variables")
    out = zeros(Float64, 1)
    HC.evaluate!(out, r.r_sys, point)
    return @inbounds out[1]
end

(r::RoutingFunction)(point::AbstractVector{<:Real}) = evaluate_r(r, point)

# ∇r(P)
function evaluate_grad_r(r::RoutingFunction, point::AbstractVector{<:Real})::Vector{Float64}
    length(point) == length(r.vars) ||
        error("length of point must match number of variables")
    out = zeros(Float64, length(r.vars))
    HC.evaluate!(out, r.grad_sys, point)
    return out
end

# g(P). g is a shifted sum of squares, so evaluating it directly beats going
# through the interpreter.
function evaluate_g(r::RoutingFunction, point::AbstractVector{<:Real})::Float64
    s = 1.0
    @inbounds for i in eachindex(r.c)
        s += (point[i] - r.c[i])^2
    end
    return s
end
