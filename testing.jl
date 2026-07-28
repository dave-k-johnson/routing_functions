using Revise
using ConnectedComponents

# 2 circles
@var x[1:2]
r = RoutingFunction(-1*one(Expression), x[1:2])
G = [(x[1]^2 + x[2]^2 - 1)*(x[1]^2 + x[2]^2 - 9)]

M, routPoints = find_connectivity_matrix(r, G; grad_step_size = 1e-1, tol = 2e-1, start_step_size = 5e-1)

# you can also do this step-by-step. build the cache once and hand it to every
# routine -- it holds the compiled systems and the scratch buffers, so passing r
# and G separately instead rebuilds all of that on every call.
cache = RoutingCache(r, G)
routPoints = routing_points(cache)
index_dict = sort_routing_points_by_index(cache, routPoints)
final_points = index_dict[0]
initial_points = vcat([v for (k,v) in index_dict if k != 0]...)

# gradient takes 2 saddles to distinct index 0 routing points, so there are 2 connected components
solns1 = solve_ivp(cache, initial_points[1], final_points; Verbose = true)
solns2 = solve_ivp(cache, initial_points[2], final_points; Verbose = true)



# elliptic curve
@var x[1:2]
r = RoutingFunction(one(Expression), x[1:2], [0.855, -1.632])
G = [x[2]^2 - x[1]*(x[1] - 1)*(x[1] + 1)]

M, routPoints = find_connectivity_matrix(r, G; grad_step_size = 1e-1, tol = 2e-1, start_step_size = 5e-1)






# elliptic curve with 2 points of distance 3 away from origin removed
@var x[1:2]
r = RoutingFunction(x[1]^2 + x[2]^2 - 9, x[1:2], [0.855, -1.632])
G = [x[2]^2 - x[1]*(x[1] - 1)*(x[1] + 1)]

M, routPoints = find_connectivity_matrix(r, G; grad_step_size = 1e-1, tol = 2e-1, start_step_size = 5e-1)



# twisted cubic with origin removed
@var x[1:3]
r = RoutingFunction(x[1]*x[2]*x[3], x[1:3])
G = [x[1]^3 - x[3], x[1]^2 - x[2]]

M, routPoints = find_connectivity_matrix(r, G)






# compact degree 4 curve with coordinate axes removed. See "Smooth Connectivity ..." paper for picture
@var x[1:2]
r = RoutingFunction(x[1]*x[2], [1/3,1/2])
G = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1]+x[2])]
M, routPoints = find_connectivity_matrix(r, G)




# ding dong from smooth connectivity paper
# removing all singular points
@var x[1:3]
g = x[1]^2 + x[2]^2 - x[3]^2 + x[3]^3
f = sum(differentiate(g, x[1:3]).^2)
r = RoutingFunction(f, [0.7978234324, 0.6623073432, 0.2347907832])
G = [g]
cache = RoutingCache(r, G)
M, routPoints = find_connectivity_matrix(cache)

index_dict = sort_routing_points_by_index(cache, routPoints)
euler_characteristic = length(index_dict[0]) - length(index_dict[1]) + length(index_dict[2])









# checking compute_matrices works by comparing with Smooth Connectivity paper
# example 2.5a, same as paper
G = [x[1]^2 - x[2]^2]
W, V = compute_matrices(G, x[1:2], [1.0,1.0])

# example 2.5b, different from paper, but its because it chooses a different basis for tangent space
# if you use the same V_x as paper, the W matrices are the same
@var x[1:3]
G = [x[1]^2 - x[2]^2*x[3]]
W, V = compute_matrices(G, x[1:3], [1.0,1.0,1.0])
H = hessian(4*x[1]^2 + 4*x[2]^2*x[3]^2 + x[2]^4, G, x[1:3], [1.,1.,1.])

# these are matrices from paper
P = hcat((1/sqrt(2) * [1 1/3; 1 -1/3; 0 4/3]), [2/3; -2/3; -1/3])
Q = hcat(V, [2/3; -2/3; -1/3])
# make right change of basis
R = (transpose(Q) * P)[1:2,1:2]

# do change of basis to get matrices from 2.5b
WW1 = transpose(R) * W[1] * R
WW2 = transpose(R) * W[2] * R
WW3 = transpose(R) * W[3] * R

# check these are the same from paper
round.(WW1; digits = 10) == round.([0 4/27; 4/27 -16/81]; digits = 10)
round.(WW2; digits = 10) == round.([0 -4/27; -4/27 16/81]; digits = 10)
round.(WW3; digits = 10) == round.([0 -2/27; -2/27 8/81]; digits = 10)

# do the same for the Hessian
HH = transpose(R) * H * R
round.(HH; digits = 10) == round.(2/81 * [567 303; 303 127]; digits = 10)



# from section 6 in paper, Chubs, this one doesn't seem to always work
@var x[1:3]
g = x[1]^4 + x[2]^4 + x[3]^4 - (x[1]^2 + x[2]^2 + x[3]^2) + 1/2
f = sum(differentiate(g, x[1:3]).^2)
c = [0.7978234324, 0.6623073432, 0.2347907832]
r = RoutingFunction(f, x[1:3], c)
G = [g]
cache = RoutingCache(r, G)

M, routPoints = find_connectivity_matrix(cache; grad_step_size = 1e-1, tol = 2e-1, start_step_size = 5e-1)

index_dict = sort_routing_points_by_index(cache, routPoints)

# should give -8, HC.jl is not finding all critical points
euler = length(index_dict[0]) - length(index_dict[1]) + length(index_dict[2])



# for chubs, tried solving in bertini instead
include("bertiniIO.jl")
bertini_solutions = read_real_parts("chubs_bertini_run/real_finite_solutions")

bertini_solutions[1:120, 1:3]

solns = [vec(bertini_solutions[i, 1:3]) for i in 1:120 if abs(evaluate(f, x[1:3] => vec(bertini_solutions[i,1:3]))) > 1e-20]

[evaluate_grad_r(r, P) for P in solns]
