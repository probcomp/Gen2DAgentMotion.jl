using Gen

##################################################
# mixture of uniform and gaussians 2D coordinate #
##################################################

struct UniformGaussianMixtureCoord <: Distribution{Vector{Float64}}
end

const mixture_of_gaussians = UniformGaussianMixtureCoord()

function (::UniformGaussianMixtureCoord)(uniform_prob, normal_means, normal_std)
    return random(mixture_of_gaussians, uniform_prob, normal_means, normal_std)
end

function Gen.random(
        ::UniformGaussianMixtureCoord,
        uniform_prob::Float64, normal_means::Vector{Tuple{Float64,Float64}}, normal_std::Float64)
    if bernoulli(uniform_prob)
        return uniform_coord()
    else
        i = uniform_discrete(1, length(normal_means))
        pt = normal_means[i]
        @assert length(pt) == 2
        x = normal(pt[1], normal_std)
        y = normal(pt[2], normal_std)
        return Float64[x, y]
    end
end

function Gen.logpdf(
        ::UniformGaussianMixtureCoord, pt::Vector{Float64},
        uniform_prob::Float64, normal_means::Vector{Tuple{Float64,Float64}}, normal_std::Float64)
    @assert length(pt) == 2
    ls = Vector{Float64}(undef, length(normal_means) + 1)
    for i in 1:length(normal_means)
        @assert length(normal_means[i]) == 2
        mu_x, mu_y = normal_means[i]
        ls[i] = log(1 - uniform_prob) - log(length(normal_means))
        ls[i] += logpdf(normal, pt[1], mu_x, normal_std) + logpdf(normal, pt[2], mu_y, normal_std)
    end
    ls[end] = log(uniform_prob)
    return logsumexp(ls)
end

#########################
# uniform 2D coordinate #
#########################

struct UniformCoord <: Distribution{Vector{Float64}}
end

const uniform_coord = UniformCoord()

function (::UniformCoord)()
    return random(uniform_coord)
end

function Gen.random(::UniformCoord)
    x = uniform(0, 1)
    y = uniform(0, 1)
    return Float64[x, y]
end

function Gen.logpdf(::UniformCoord, pt::Vector{Float64})
    @assert length(pt) == 2
    if (0.0 <= pt[1] <= 1.0) && (0.0 <= pt[2] <= 1.0)
        return 0.0
    else
        return -Inf
    end
end

############################################
# uniform 2d coordinate within a rectangle #
############################################

struct UniformCoordRect <: Distribution{Vector{Float64}}
end

const uniform_coord_rect = UniformCoordRect()

function (::UniformCoordRect)()
    return random(uniform_coord_rect)
end

function Gen.random(::UniformCoordRect, xmin, xmax, ymin, ymax)
    x = uniform(xmin, xmax)
    y = uniform(ymin, ymax)
    return Float64[x, y]
end

function Gen.logpdf(::UniformCoordRect, pt::Vector{Float64}, xmin, xmax, ymin, ymax)
    @assert length(pt) == 2
    if (xmin <= pt[1] <= xmax) && (ymin <= pt[2] <= ymax)
        return -log(xmax - xmin) - log(ymax - ymin)
    else
        return -Inf
    end
end

Gen.has_argument_grads(::UniformCoordRect) = (false, false, false, false)
Gen.has_output_grad(::UniformCoordRect) = false

export mixture_of_gaussians, uniform_coord, uniform_coord_rect
