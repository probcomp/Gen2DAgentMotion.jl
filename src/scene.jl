import Random

struct Point
    x::Float64
    y::Float64
end

function Point(vec::Vector{Float64})
    length(vec) != 2 && error("wrong length")
    return Point(vec[1], vec[2])
end

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)
end

function dist_sq(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy
end

Base.:+(a::Point, b::Point) = Point(a.x + b.x, a.y + b.y)
Base.:*(a::Real, b::Point) = Point(b.x * a, b.y * a)

function ccw(a::Point, b::Point, c::Point)
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
end

function line_intersects_line(a1::Point, a2::Point, b1::Point, b2::Point)
    # http://algs4.cs.princeton.edu/91primitives/
    return (ccw(a1, a2, b1) * ccw(a1, a2, b2) <= 0.0) && (ccw(b1, b2, a1) * ccw(b1, b2, a2) <= 0.0)
end

struct Wall
    a::Point
    b::Point
end

struct Bounds
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

struct Scene    
    bounds::Bounds
    walls::Vector{Wall}
end

function affine(x::Real, min1, max1, min2, max2)
    offset = (x - min1) / (max1 - min1)
    return min2 + offset * (max2 - min2)
end

function transform(point::Point, prev_bounds::Bounds, new_bounds::Bounds)
    new_x = affine(point.x, prev_bounds.xmin, prev_bounds.xmax, new_bounds.xmin, new_bounds.xmax)
    new_y = affine(point.y, prev_bounds.ymin, prev_bounds.ymax, new_bounds.ymin, new_bounds.ymax)
    return Point(new_x, new_y)
end

function transform(wall::Wall, prev_bounds::Bounds, new_bounds::Bounds)
    return Wall(transform(wall.a, prev_bounds, new_bounds), transform(wall.b, prev_bounds, new_bounds))
end


function resize(scene::Scene, new_bounds::Bounds)
    new_walls = map((wall) -> transform(wall, scene.bounds, new_bounds), scene.walls)
    return Scene(new_bounds, new_walls)
end

function intersects(wall::Wall, a::Point, b::Point)
    return line_intersects_line(wall.a, wall.b, a, b)
end

function line_of_site(scene::Scene, a::Point, b::Point)
    for wall in scene.walls
        if intersects(wall, a, b)
            return false
        end
    end
    return true
end

function example_apartment_floorplan()
    door_width = 0.4

    left_wall = Wall(Point(3.0, 0.63), Point(3.0, 4.27))
    right_wall = Wall(Point(6.78, 1.19), Point(6.78, 4.27))
    bottom_wall = Wall(Point(3.0, 4.27), Point(6.78, 4.27))

    top_wall1 = Wall(Point(3.0, 0.63), Point(4.64, 0.63)) # horizontal
    top_wall2 = Wall(Point(4.64, 0.63), Point(4.64, 1.19)) # vertical
    top_wall3 = Wall(Point(4.64, 1.19), Point(6.78, 1.19)) # horizontal

    internal_wall1 = Wall(Point(5.50, 1.19), Point(5.50, 2.10 - door_width)) # vertical with door, bedroom wall

    internal_wall2 = Wall(Point(5.50, 2.10), Point(5.50, 3.03)) # vertical, closet wall
    internal_wall3 = Wall(Point(5.50, 2.10), Point(6.78, 2.10)) # wall between bedroom and closet

    internal_wall4 = Wall(Point(5.50, 3.03), Point(5.76, 3.03)) # short horizontal wall
    internal_wall5 = Wall(Point(5.76, 3.03 + door_width), Point(5.76, 3.61))
    internal_wall6_part1 = Wall(Point(5.76, 3.61), Point(6.20, 3.61)) # angled.
    internal_wall6_part2 = Wall(Point(6.20, 3.87), Point(6.20, 3.61)) # angled.
    internal_wall7 = Wall(Point(6.20, 3.87), Point(6.20, 4.27))
    internal_wall8 = Wall(Point(3.0 + door_width, 2.88), Point(5.04, 2.88)) # horizontal wall
    internal_wall9 = Wall(Point(3.92, 2.88), Point(3.92, 3.61 - door_width))
    internal_wall10 = Wall(Point(3.92, 4.27), Point(3.92, 3.61 + door_width)) 
    internal_wall11 = Wall(Point(3.92, 3.61), Point(5.31, 3.61)) # horizontal to hallway
    internal_wall12 = Wall(Point(5.31, 3.61), Point(5.31, 4.27)) # hallway vertical
    internal_wall13 = Wall(Point(5.04, 2.88), Point(5.04, 3.61)) # hallway vertical

    counter = Wall(Point(3.5, 2.12), Point(4.53, 2.12))
    walls = [
        left_wall,
        right_wall,
        top_wall1,
        top_wall2,
        top_wall3,
        bottom_wall,
        internal_wall1,
        internal_wall2,
        internal_wall3,
        internal_wall4,
        internal_wall5,
        internal_wall6_part1,
        internal_wall6_part2,
        internal_wall7,
        internal_wall8,
        internal_wall9,
        internal_wall10,
        internal_wall11,
        internal_wall12, 
        internal_wall13,
        counter
    ]
    scene = Scene(Bounds(3.0, 6.78, 4.27, 0.63), walls)
    scene = resize(scene, Bounds(0.0, 1.0, 0.0, 1.0))
    return scene
end

export Point, Wall, Bounds, Scene, resize
