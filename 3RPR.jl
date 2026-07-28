using ConnectedComponents
using LinearAlgebra

a = [0, 14, 7] #.*0.1
b = [0, 0, 10] #.*0.1
A = [0, 16, 9] #.*0.1
B = [0, 0, 6] #.*0.1
c3 = 100 #0.1^2
P1 = [0,0] #.*0.1
P2 = [a[2],0] #.*0.1
P3 = [a[3],b[3]] #.*0.1



@var p[1:2], ϕ[1:2], c[1:2]

F = [
    ϕ[1]^2 + ϕ[2]^2 - 1,
    p[1]^2 + p[2]^2 - 2*(a[3]*p[1] + b[3]*p[2])*p[1] + 2*(b[3]*p[1] - a[3]*p[2])*ϕ[2] + a[3]^2 + b[3]^2 - c[1],
    p[1]^2 + p[2]^2 - 2*A[2]*p[1] + 2*((a[2]-a[3])*p[1] - b[3]*p[2] + A[2]*a[3] - A[2]*a[2])*ϕ[1] + 2*(b[3]*p[1]+(a[2]-a[3])*p[2] - A[2]*b[3])*ϕ[2] + (a[2]-a[3])^2 + b[3]^2 + A[2]^2 - c[2],
    p[1]^2 + p[2]^2 - 2*(A[3]*p[1] + B[3]*p[1]) + A[3]^2 + B[3]^2 - c3
]

vars = vcat(p[1:2], ϕ[1:2], c[1:2])

JF = differentiate(F, vars)

r = RoutingFunction(det(JF[1:4,1:4]), vars)

# build the cache once: every routine below reuses its compiled systems and buffers
cache = RoutingCache(r, F)

# this system carries constants of size A[2] = 16 and c3 = 100, so the part of
# V(F) we care about is nowhere near the default [-3,3]^6 search box and almost no
# random start projects onto the variety. 20 gets the hit rate up to roughly 1/3.
routPoints = routing_points(cache; box = 20.0, verbose = true)

index_dict = sort_routing_points_by_index(cache, routPoints)

initial_points = vcat([v for (k,v) in index_dict if k > 0]...)
final_points = index_dict[0]

for P in initial_points
    solve_ivp(cache, P, final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
end




M, routPoints = find_connectivity_matrix(cache; grad_step_size = 1.0, tol = 2.0,
                                         start_step_size = 1.1, box = 20.0, verbose = true)

