# Gen2DAgentMotion.jl

This Julia package contains Gen modeling components for building generative
models of the motion of agents that move approximately rationally around a 2D
environment with obstacles.

You can combine the modeling components in this package with your own Gen
models of the agent's tasks.

The repository also contains a minimal example that uses these components.

Author: [Marco Cusumano-Towner](https://www.mct.dev)

## Simple 2D scenes

The package exports some types used to define simple 2D scenes with obstacles.

- `Point(x::Float64, y::Float64)`: A 2D point in the scene.

- `Bounds(xmin::Float64, xmax, ymin, ymax)`: A rectangular bounding box for the scene.

- `Wall(a::Point, b::Point)`: A line segment representing an impassable obstacle.

- `Scene(bounds::Bounds, walls:Vector{Wall})`: Construct a scene.

## Path planner model

A path planner based on [rapidly exploring random trees](http://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf) (RRTs),
followed by simple trajectory optimization.

- `PlannerParams(rrt_iters::UInt, rrt_dt::Float64, refine_iters::UInt, refine_std::Float64)`

- `(path, failed, tree) = plan_and_optimize_path(scene::Scene, a::Point, b::Point; params::PlannerParams)`: Plan a path from point `a` to point `b`, avoiding obstacles in the scene. Return a path (a `Vector{Point}`), a `Bool` indicating if a path was found or not, and the RRT that was used internally for debugging purposes. If the path planning failed, then the path is a 0-element vector.

## Observation model



- `ObsModelParams`

- `path_observation_model`

## Minimal example

See `examples/minimal/`

![Inferences from minimal example](/examples/minimal/inferences.png)
