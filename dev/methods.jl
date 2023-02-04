# methods.jl
# Collection of coupling and energy functions

function none_method(v)
    return 0 * v  # for the type stability thing
end

function none_method(v1, v2)
    return 0 * v1  # for the type stability thing
end

const Pi2 = pi / 2
const Pi4 = pi / 4

"""
    cosine(v)

The coupling energy inducing the XY-model (rank-2 relaxation).
"""
function cosine(v)
    return cos.(Pi2 .* v)
end

function sine(v)
    return sin.(Pi2 .* v)
end

function sine(v1, v2)
    return sine(v1 - v2)
end

function linear(v1, v2)
    return v1 - v2
end

function dtri(v)
    # the derivative of the triangular function
    p = Int.(floor.(v ./ 2 .+ 1 / 2))
    parity = rem.(abs.(p), 2)
    return -2 .* parity .+ 1
end

function triangular(v)
    # Odd piece-wise linear defined within the period [-2, 2] by
    # phi(-2) = phi(0) = phi(2) = 0
    # - phi(-1) = phi(1) = 1
    # This works for both scalars and vectors 
    p = Int.(floor.(v ./ 2 .+ 1 / 2))
    parity = rem.(abs.(p), 2)
    return (v .- p .* 2) .* (-2 .* parity .+ 1)
end

function triangular_mod(v)
    # Odd piece-wise linear defined within the period [-2, 2] by
    # phi(-2) = phi(0) = phi(2) = 0
    # - phi(-1) = phi(1) = 1
    # this is slightly slower than triangular(v) above because of the mod
    # but looks easier to understand
    vr = mod.(v .+ 1, 4) .- 1
    return vr .+ min.(0 .* vr, -vr .+ 1).*2
end

function triangular(v1, v2)
    return triangular(v1 - v2)
end

const pwDELTA = 0.1
# const pwPERIOD = 4.0

"""
    piecewise_generic(v, Delta = 0.1)

Evaluate piece-wise linear odd periodic ([-2, 2]) skewed triangular
function of the vector argument `v` that equals to 0 at 0 and 2 and
to 1 at 1 + Delta.
"""
function piecewise_generic(v, Delta = 0.1)
    # This version is slow but generic
    vbar = mod.(v .+ 2, 4) .- 2
    s = sign.(vbar)
    vbabs = s.*vbar

    ind = vbabs .> 1 + Delta
    out = vbabs./(1 + Delta) - 2 .* ind.*(vbabs .- (1 + Delta))./(1 - Delta^2)

    return s .* out
end

function piecewise_local(v, Delta = 0.1)
    # this version relies on the small variation of v
    # TODO: make it decent
    vabs = abs(v)
    out =
        if vabs < 1 + Delta
            sign(v)/(1+Delta)*vabs
        else
            if vabs < 3 - Delta
                -sign(v)/(1 - Delta)*(vabs - 2)
            else
                if vabs < 5 + Delta
                    sign(v)/(1+Delta)*(vabs - 4)
                else
                    # add one more step?
                    piecewise_generic(v, Delta)
                end
            end
        end
    return out
 end

function piecewise(v1, v2, Delta = 0.1)
    return piecewise_local(v1 - v2, Delta)
end

function square(v)
    return dtri(v .+ 1)
end

function square(v1, v2)
    return square(v1 - v2)
end

function bilinear(v1, v2)
    return -dtri(v1) .* triangular(v2)
end

function bisine(v1, v2)
    return -Pi2 .* cos(Pi2 .* v1) .* triangular(Pi2 .* v2)
end

function squarishk(v, k=10)
    return tanh.(k .* triangular(v))
end

function squarish(v1, v2, k=10)
    return squarishk(v1 - v2, k)
end

# this really should be a model parameter
const stW = 0.55 * 2

function skew_triangular(v)
    c1 = 1 / (stW * (stW - 2))

    vbar = mod.(v .+ 2, 4) .- 2
    s = sign.(vbar)
    svbar = s .* vbar
    ind = sign.(svbar .- stW)

    mid = (svbar .* (stW - 1) .- stW) .* c1
    Delta = (svbar .- stW) .* c1

    out = Delta .* ind .+ mid
    return s .* out
end

function skew_triangular(v1, v2)
    return skew_triangular(v1 - v2)
end

#
# Continuous representation models
#

function continuous_model_1(v1, v2)
    return -dtri(v1).*triangular(v2)
end

function dtri_cont(v, s)
    # scalar version only
    vreduced = mod(v + 2, 4) - 2
    vabs = abs(vreduced)
    m = 2/(2 - s)
    out =
        if vabs < 1 - s
            m
        elseif vabs <= 1 + s
            (1 - vabs)*m/s
        else
            -m
        end
    return out
end

function continuous_model_1_cont(v1, v2, s = 0.1)
    # change rate in Model 1 with continuous derivative
    # s is the half-width of the transitional region between -1 and 1
    # v1 is presumed to be scalar
    # noticeably slower
    return -dtri_cont(v1, s).*triangular(v2)
end

#
# Models for the separated representation
#

function rank_2_SDP_dynamic_2(v)
    # the coupling function of rank-2 SDP in the separated representation
    # 
    return PI2 * sin(PI2*v)
end

function rank_2_SDP_static_2(v)
    # the coupling function of rank-2 SDP in the separated representation
    # 
    return sin(PI4*v)^2
end

# For model 2 (rank-2 GW-representation), the coupling function is
# sign(v)/2 for v \in [-2,2]
function continuous_model_2(v)
    vreduced = mod(v + 2, 4) - 2
    return sign(vreduced)/2
end

function continuous_model_2(v1, v2)
    return continuous_model_2(v1 - v2)
end

# The cut counting function for model II
# For v in [-2, 2], Phi(v) = |v|/2
function continuous_model_2_energy(v)
    vreduced = mod(v + 2, 4) - 2
    return abs(vreduced)/2
end

function continuous_model_2_energy(v1, v2)
    return continuous_model_2_energy(v1 - v2)
end

function coupling_model_2(x1, x2, gamma = 0.0)
    dd = x1 - x2
    return sign(dd) + gamma*dd
end


## Noise generators

function noiseUniform(L::Int)::Array{Float64, 1}
    # uniform distribution in interval [-1, 1]
    # returns vector of length L
    return rand(Uniform(-1, 1), L)
end

function noiseNormal(L::Int)::Array{Float64, 1}
    # normal distribution with zero mean and unit variance
    # returns vector of length L
    return rand(Normal(0, 1), L)
end
