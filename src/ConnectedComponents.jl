module ConnectedComponents

using HomotopyContinuation, LinearAlgebra, SciMLBase, OrdinaryDiffEq

const HC = HomotopyContinuation
const LA = LinearAlgebra

using Reexport: @reexport
@reexport using HomotopyContinuation

include("routing_functions.jl")
include("cache.jl")
include("routing_points.jl")
include("hessian.jl")
include("path_tracking.jl")
include("connectivity.jl")

export RoutingFunction,
    RoutingCache,
    evaluate_r,
    evaluate_grad_r,
    evaluate_g,
    evaluate_G!,
    jacobian_G!,
    routing_system,
    routing_points,
    flow_to_routing_points,
    ambient_gradient_hessian,
    compute_matrices,
    hessian,
    hessian_and_tangent,
    idx,
    sort_routing_points_by_index,
    project_to_variety!,
    distance_to_endpoints,
    nearest_index,
    projected_gradient_field,
    gradient_flow!,
    find_starting_points_for_flow,
    solve_ivp,
    find_connectivity_matrix

end
