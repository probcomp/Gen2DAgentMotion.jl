# depends on scene.jl

##############################################
# rapidly exploring random tree for 2D point #
##############################################

mutable struct RRTTree
    confs::Matrix{Float64} # 2 x max_num_nodes
    control_xs::Vector{Float64}
    control_ys::Vector{Float64}
    costs_from_start::Vector{Float64}
    parents::Vector{UInt}
    num_nodes::UInt
    max_num_nodes::UInt
end

function RRTTree(root_conf::Point, max_num_nodes::UInt)

    # pre-allocate
    confs = Matrix{Float64}(undef, 2, max_num_nodes)
    control_xs = Vector{Float64}(undef, max_num_nodes)
    control_ys = Vector{Float64}(undef, max_num_nodes)
    costs_from_start = Vector{Float64}(undef, max_num_nodes)
    parents = Vector{UInt}(undef, max_num_nodes)

    # initialize
    confs[1,1] = root_conf.x
    confs[2,1] = root_conf.y
    control_xs[1] = NaN
    control_ys[1] = NaN
    costs_from_start[1] = 0.0
    parents[1] = 0
    
    return RRTTree(confs, control_xs, control_ys, costs_from_start, parents, 1, max_num_nodes)
end

@inline function add_node!(tree::RRTTree, parent::UInt, new_conf::Point, control::Point, cost_from_start::Float64)
    node = tree.num_nodes + 1
    tree.confs[1,node] = new_conf.x
    tree.confs[2,node] = new_conf.y
    tree.control_xs[node] = control.x # TODO fix make dense..
    tree.control_ys[node] = control.y
    tree.costs_from_start[node] = cost_from_start
    tree.parents[node] = parent
    tree.num_nodes = node
    return node
end

function nearest_neighbor(conf::Point, confs::Matrix{Float64}, num_nodes::UInt)
    nearest::UInt = 1
    @inbounds dx::Float64 = confs[1,1] - conf.x
    @inbounds dy::Float64 = confs[2,1] - conf.y
    min_so_far::Float64 = dx * dx + dy * dy
    local d::Float64
    for node in 2:num_nodes
        @inbounds dx = confs[1,node] - conf.x
        @inbounds dy = confs[2,node] - conf.y
        d = dx * dx + dy * dy
        if d < min_so_far
            nearest = node
            min_so_far = d
        end
    end
    return nearest
end

struct SelectControlResult
    start_conf::Point
    new_conf::Point
    control::Point
    failed::Bool # new_conf is undefined in this case
    cost::Float64 # cost of this control action (e.g. distance)
end

@inline function random_config(bounds)
    x = rand()* (bounds.xmax - bounds.xmin) + bounds.xmin
    y = rand() * (bounds.ymax - bounds.ymin) + bounds.ymin
    return Point(x, y)
end

function select_control(walls, target_conf::Point, start_conf::Point, dt::Float64)

    dist_to_target = dist(start_conf, target_conf)
    diff = Point(target_conf.x - start_conf.x, target_conf.y - start_conf.y)
    distance_to_move = min(dt, dist_to_target)
    scale = distance_to_move / dist_to_target
    control = Point(scale * diff.x, scale * diff.y)

    # go in the direction of target_conf from start_conf 
    new_conf = Point(start_conf.x + control.x, start_conf.y + control.y)

    # test the obstacles
    failed = false
    for wall in walls
        if line_intersects_line(wall.a, wall.b, start_conf, new_conf)
            failed = true
            break
        end
    end
    cost = distance_to_move
    return SelectControlResult(start_conf, new_conf, control, failed, cost)
end


function rrt(bounds::Bounds, walls::Vector, init::Point, iters::UInt, dt::Float64)
    tree = RRTTree(init, iters)
    local near_node::UInt
    for iter=1:iters
        rand_conf::Point = random_config(bounds)
        near_node = nearest_neighbor(rand_conf, tree.confs, tree.num_nodes)
        result = select_control(walls, rand_conf, Point(tree.confs[1,near_node], tree.confs[2,near_node]), dt)
        if !result.failed
            cost_from_start = tree.costs_from_start[near_node] + result.cost
            add_node!(tree, near_node, result.new_conf, result.control, cost_from_start)
        end
    end
    return tree
end

##############################
# path planner that uses RRT #
##############################

struct PlannerParams
    rrt_iters::UInt
    rrt_dt::Float64 # the maximum proposal distance
    refine_iters::UInt
    refine_std::Float64
end

struct Path
    start::Point
    goal::Point
    points::Array{Point,1}
end

function concatenate(a::Path, b::Path)
    if a.goal.x != b.start.x || a.goal.y != b.start.y
        error("goal of first path muts be start of second path")
    end
    points = Array{Point,1}()
    for point in a.points
        push!(points, point)
    end
    for point in b.points[2:end]
        push!(points, point)
    end
    @assert points[1].x == a.start.x
    @assert points[1].y == a.start.y
    @assert points[end].x == b.goal.x
    @assert points[end].y == b.goal.y
    Path(a.start, b.goal, points)
end

function add_intermediate_points!(new_points, next_point, spacing)
    # add intermediate points, evently spaced
    d = dist(next_point, new_points[end])
    while d > spacing # add points
        intermediate_point = ((d-spacing)/d) * new_points[end] + (spacing/d) * next_point
        @assert isapprox(dist(intermediate_point, new_points[end]), spacing)
        push!(new_points, intermediate_point)
        d = dist(next_point, new_points[end])
    end
end

"""
Remove intermediate nodes that are not necessary, so that refinement
optimization is a lower-dimensional optimization problem.
"""
function simplify_path(scene::Scene, original::Path; spacing=0.05)
    new_points = Array{Point,1}()
    push!(new_points, original.start)
    for i=2:length(original.points) - 1

        # try skipping this point
        if !line_of_site(scene, new_points[end], original.points[i + 1])
            add_intermediate_points!(new_points, original.points[i], spacing)
            push!(new_points, original.points[i])
        end

        # then we can skip it?
    end
    @assert line_of_site(scene, new_points[end], original.goal)
    add_intermediate_points!(new_points, original.goal, spacing)
    push!(new_points, original.goal)
    return Path(original.start, original.goal, new_points)
end

function local_score(scene, a, b, num_tests, perturb_width)
    score = 0.
    for i in 1:num_tests
        perturbed_a = Point(a.x + uniform(-perturb_width, perturb_width), a.y + uniform(-perturb_width, perturb_width))
        perturbed_b = Point(b.x + uniform(-perturb_width, perturb_width), b.y + uniform(-perturb_width, perturb_width))
        score += line_of_site(scene, perturbed_a, perturbed_b)
    end
    return score / num_tests
end

"""
Optimize the path to minimize its length while avoiding obstacles in the scene.
"""
function refine_path(
        scene::Scene, original::Path, iters::UInt, std::Float64, dist_weight, num_samples, perturb_width)

    # do stochastic optimization
    new_points = deepcopy(original.points)
    num_interior_points = length(original.points) -2
    if num_interior_points == 0
        return original
    end
    gap = 0.05 # TODO
    for i=1:iters
        point_idx = 2 + (i % num_interior_points)
        @assert point_idx > 1 # not start
        @assert point_idx < length(original.points) # not goal
        prev_point = new_points[point_idx-1]
        point = new_points[point_idx]
        next_point = new_points[point_idx+1]
        adjusted = Point(point.x + randn() * std, point.y + randn() * std)
        old_dist_sq = dist_sq(prev_point, point) + dist_sq(point, next_point)
        new_dist_sq = dist_sq(prev_point, adjusted) + dist_sq(adjusted, next_point)

        cur_1 = Point(point.x + gap, point.y)
        cur_2 = Point(point.x - gap, point.y)
        cur_3 = Point(point.x, point.y + gap)
        cur_4 = Point(point.x, point.y - gap)

        adjusted_1 = Point(adjusted.x + gap, adjusted.y)
        adjusted_2 = Point(adjusted.x - gap, adjusted.y)
        adjusted_3 = Point(adjusted.x, adjusted.y + gap)
        adjusted_4 = Point(adjusted.x, adjusted.y - gap)

        cur_cost = 0.001 * old_dist_sq
        adjusted_cost = 0.001 * new_dist_sq
        for wall in scene.walls
            
            cur_cost += (
                intersects(wall, prev_point, cur_1) +
                intersects(wall, cur_1, next_point) +
                intersects(wall, prev_point, cur_2) +
                intersects(wall, cur_2, next_point) +
                intersects(wall, prev_point, cur_3) +
                intersects(wall, cur_3, next_point) +
                intersects(wall, prev_point, cur_4) +
                intersects(wall, cur_4, next_point))

            adjusted_cost += (
                intersects(wall, prev_point, adjusted_1) +
                intersects(wall, adjusted_1, next_point) +
                intersects(wall, prev_point, adjusted_2) +
                intersects(wall, adjusted_2, next_point) +
                intersects(wall, prev_point, adjusted_3) +
                intersects(wall, adjusted_3, next_point) +
                intersects(wall, prev_point, adjusted_4) +
                intersects(wall, adjusted_4, next_point))
        end
        if adjusted_cost < cur_cost
            # buffer scores between 0 and 4
            new_points[point_idx] = adjusted
        end
    end
    Path(original.start, original.goal, new_points)
end

function optimize_path(
        scene::Scene, original::Path,
        refine_iters::UInt, refine_std::Float64, spacing::Float64,
        dist_weight, num_samples, perturb_width)
    simplified = simplify_path(scene, original; spacing=spacing)
    refined = refine_path(scene, simplified, refine_iters, refine_std, dist_weight, num_samples, perturb_width)
    return refined 
end

@inline function get_conf_point(tree, node::UInt)
    return Point(tree.confs[1,node], tree.confs[2,node])
end

"""
Plan path from start to goal that avoids obstacles in the scene.
"""
function plan_path(
        start::Point, goal::Point, scene::Scene,
        params::PlannerParams=PlannerParams(2000, 3.0, 10000, 1.))

    tree = rrt(scene.bounds, scene.walls, start, params.rrt_iters, params.rrt_dt)

    # find the best path along the tree to the goal, if one exists
    best_node = 0
    min_cost = Inf
    path_found = false
    best_conf = Point(NaN,NaN)
    for node in 1:tree.num_nodes
        # check for line-of-site to the goal
        conf = get_conf_point(tree, node)
        clear_path = line_of_site(scene, conf, goal)
        cost = tree.costs_from_start[node] + (clear_path ? dist(conf, goal) : Inf)
        if cost < min_cost
            path_found = true
            best_node = node
            min_cost = cost
            best_conf = conf
        end
    end

    local path::Union{Nothing,Path}
    if path_found
        # extend the tree to the goal configuration
        control = Point(goal.x - best_conf.x, goal.y - best_conf.y) 
        goal_node = add_node!(tree, best_node, goal, control, min_cost)
        points = Array{Point,1}()
        node = goal_node
        push!(points, get_conf_point(tree, goal_node))
        # the path will contain the start and goal
        while node != 1
            node = tree.parents[node]
            push!(points, get_conf_point(tree, node))
        end
        @assert points[end] == start # the start point
        @assert points[1] == goal
        path = Path(start, goal, reverse(points))
    else
        path = nothing
    end
    
    return path, tree
end

function plan_and_optimize_path(scene, prev_loc, loc, params)
    (path, tree) = plan_path(prev_loc, loc, scene, params)
    if isnothing(path)
        return Point[], true, tree
    else
        spacing = 0.2
        dist_weight = 5.0
        num_samples = 5
        perturb_width = 0.05
        path = optimize_path(scene, path, params.refine_iters, params.refine_std, spacing,
            dist_weight, num_samples, perturb_width)
        points = path.points
        @assert points[1] == prev_loc
        @assert points[end] == loc
        return points[2:end], false, tree
    end
end

export PlannerParams, plan_path, plan_and_optimize_path
