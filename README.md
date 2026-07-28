# ConnectedComponents.jl

**ConnectedComponents.jl** is a Julia package for studying the **connectivity of real algebraic varieties** using *routing functions*.  

The package implements algorithms from  
> **"Smooth Connectivity in Real Algebraic Varieties"**  
> *Joseph Cummings, Jonathan Hauenstein, Hoon Hong, and Clifford Smyth*
> *Numerical Algorithms, 100(1), 63-84, 2025*

This paper introduces the use of **routing functions** — rational functions whose gradient flows reveal the connected components of a real algebraic variety.  
`ConnectedComponents.jl` automates this process, providing an efficient framework for computing critical points and tracking gradient paths.

Several examples are included in 'testing.jl'. 

## Usage

```julia
using ConnectedComponents

@var x[1:2]
r = RoutingFunction(x[1]*x[2], [1/3, 1/2])              # r = f / g^d
G = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1] + x[2])] # the variety V(G)

M, routPoints = find_connectivity_matrix(r, G)
```

Every routine also accepts a `RoutingCache`, which precomputes the symbolic
derivatives, compiles them into `HomotopyContinuation.InterpretedSystem`s, builds
the routing system and the parametrised family monodromy runs over, and allocates
the scratch buffers the numerical kernels write into:

```julia
cache = RoutingCache(r, G)
routPoints  = routing_points(cache)
index_dict  = sort_routing_points_by_index(cache, routPoints)
solns       = solve_ivp(cache, index_dict[1][1], index_dict[0])
```

Passing `r` and `G` separately builds a throwaway cache on every call, so do that
only for one-off work. A cache carries mutable buffers and must not be shared
across threads -- build one per thread.

### Badly scaled systems

The routing points are seeded by flowing from random points of `[-box, box]^n`
projected onto `V(G)`. If your equations carry large constants, the default
`box = 3.0` misses the variety entirely and almost nothing projects onto it.
`flow_to_routing_points` discards starts that fail to land (they cost a full ODE
integration and contribute nothing) and warns when it cannot fill `nstarts`:

```
┌ Warning: flow_to_routing_points: 6 of 10 starts landed on V(G) in 200 attempts
│ (box = 3.0); raise `box` or `max_attempts`
```

`box`, `nstarts`, `proj_tol` and `max_attempts` pass through `routing_points` and
`find_connectivity_matrix`. For the 3RPR example (constants of size 16 and 100)
`box = 20.0` takes the hit rate from ~0 to roughly 1 in 5, which is the difference
between the flow stage not finishing and taking about a second.

Comments and suggestions are welcome!

