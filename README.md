# Gen2DAgentMotion.jl

[![Build Status](https://travis-ci.com/probcomp/Gen2DAgentMotion.jl.svg?branch=master)](https://travis-ci.com/probcomp/Gen2DAgentMotion.jl)


This Julia package contains [Gen.jl](https://www.gen.dev) modeling components for building generative
models of the motion of agents that move approximately rationally around a 2D
environment with obstacles.

You can combine the modeling components in this package with your own Gen
models of the agent's tasks.

The repository also contains a minimal example that uses these components.

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

The generative function internally marginalizes over the random choices governing the agent's motion along its path using dynamic programming. (Conceptually, this is similar to [dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping), but computes the sum over all possible alignments between the observed and simulated trajectories, instead of the most likely alignment).

- `ObsModelParams(nominal_speed::Float64, walk_noise::Float64, noise::Float64)`: Parameters for the observation model. `nomimal_speed` is the nominal speed of the agent of the agent as it walks along its piecewise linear path. `walk_noise` is a number between `0.0` (exclusive) and `1.0` (exclusive) that governs how much the agent is expected to deviate from its nominal speed (a value of `0.2` is a reasonable starting point).

- `path_observation_model`: The generative function representing the observation model. Takes arguments of the form `(path::Vector{Point}, obs_times::Vector{Float64}, params::ObsModelParams)`, and samples addresses `(:x, 1)`, `(:y, 1)`, ..., `(:x, T)`, `(:y, T)` where `T = length(obs_times)`.

## Installing

This package is not currently available on the Julia general package registry. Instead from the Julia Pkg REPL, run:
```
add https://github.com/probcomp/Gen2DAgentMotion.jl
```

## Minimal example

The repository contians a minimal example that uses the components above, in `examples/minimal/run.jl`.
The example places a uniform prior distribution on the destination of the agent within a known floorplan.
Given a known starting location, and a set of noisy observations, the algorithm uses Gen's importance sampling
to perform probabilistic inference over the destination of the agent.
Below are images generated from the minimal example for four different observation sequences.
Observed locations are shown in black and conditional posterior samples of the destination are shown in red:

![Inferences from minimal example](/examples/minimal/inferences.png)

You can run the example by entering the `examples/minimal` directory, and running
```
julia --project=. run.jl
```

## Using in more complex models

These modeling components can be used to construct more complex models of agents that are performing tasks or trying to achieve goals that involve multiple steps.
For example, you can specify a custom prior distribution on over a sequence of waypoints that the agent visits in pursuit of their goals, and then concatenate paths returned from `plan_and_optimize_path` for each consecutive pair of waypoints, and then model observations along the resulting path with `path_observation_model`.
Thus, this package serves as a bridge between symbolic probabilistic models of a task domain and geometric models of motion through continuous spaces that include obstacles.

Many other modeling variants are possible. For example, whereas in the minimal example the scene (the obstacles) is assumed known, the scene itself can have a prior distribution and be inferred from the observed movement data.
Also, the parameters of the observation model can be inferred from data as well.

## Citing

If you use this package in your research, please cite the following paper:

Cusumano-Towner, Marco F., et al. "Probabilistic programs for inferring the goals of autonomous agents." arXiv preprint arXiv:1704.04977 (2017).

```
@article{cusumano2017probabilistic,
  title={Probabilistic programs for inferring the goals of autonomous agents},
  author={Cusumano-Towner, Marco F and Radul, Alexey and Wingate, David and Mansinghka, Vikash K},
  journal={arXiv preprint arXiv:1704.04977},
  year={2017}
}
````
