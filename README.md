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

The package also includes a path planner based on [rapidly exploring random trees](http://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf) (RRTs),
followed by simple trajectory optimization.

- `PlannerParams(rrt_iters::UInt, rrt_dt::Float64, refine_iters::UInt, refine_std::Float64)`

- `(path, failed, tree) = plan_and_optimize_path(scene::Scene, a::Point, b::Point, params::PlannerParams)`: Plan a path from point `a` to point `b`, avoiding obstacles in the scene. Return a path (a `Vector{Point}`), a `Bool` indicating if a path was found or not, and the RRT that was used internally for debugging purposes. If the path planning failed, then the path is a 0-element vector.

## Observation model

The package also includes an observation model for the movement of the agent along a path.
This is a generative model of the measured location of the agent at a set of given time steps.
The generative model is implemented as a Gen generative function, and takes as input
the path (a `Vector{Point}`) which defines a piecewise linear path through the scene,
the set of time points to measure the agent's location,
and parameters, which include a nominal speed of the agent along its path,
and noise parameters.

The observation model uses two types of noise:
variability in the agent's movement along its path, and
Gaussian measurement noise.
First, the model simulates the agent walking along the given path at its nominal speed,
and records the location of the agent at each of the to-be-observed time points.
Then, the model steps through these locations according to a random process, where there some probability that a location will be skipped, some probability that the agent will stay at the previous location, and some probability that the agent will advance to the next location.
This processes allows the model to be robust to variability in the speed of the agent as it moves along its path.
Finally, after the paths along the path have been generated,
the model adds independent isotropic Gaussian noise to each location to generate the observed locations.

The generative function internally marginalizes over the random choices governing the agent's motion along its path using dynamic programming. (Conceptually, this is similar to [dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping), but computes the sum over all possible alignments of the observed and simulated trajectories, instead of the most likely alignment).

- `ObsModelParams(nominal_speed::Float64, walk_noise::Float64, noise::Float64)`: Parameters for the observation model. `nomimal_speed` is the nominal speed of the agent of the agent as it walks along its piecewise linear path. `walk_noise` is a number between `0.0` (exclusive) and `1.0` (exclusive) that governs how much the agent is expected to deviate from its nominal speed (a value of `0.2` is a reasonable starting point).

- `path_observation_model`: The generative function representing the observation model. Takes arguments of the form `(path::Vector{Point}, obs_times::Vector{Float64}, params::ObsModelParams)`, and samples addresses `(:x, 1)`, `(:y, 1)`, ..., `(:x, T)`, `(:y, T)` where `T = length(obs_times)`.

## Minimal example

See `examples/minimal/`

![Inferences from minimal example](/examples/minimal/inferences.png)
