# Literal forward-sensitivity implementation of the Friberg--Karlsson FOCEI
# derivative.  Unlike the AD comparators, this path propagates the original
# eight model states together with analytic first- and second-order state
# sensitivity equations.  The model-specific RHS coefficients were generated
# by symbolic differentiation at build time; the run-time path is Float64-only.

include(joinpath(@__DIR__, "generated_fk_sensitivity_derivatives.jl"))

const FK_SENS_U = 16                       # 9 structural theta + 7 eta directions
const FK_SENS_D = 27                       # all 20 population + 7 EBE directions
const FK_SENS_ETA_U = 10:16
const FK_SENS_PAIRS = let pairs = Tuple{Int,Int}[]
    for a in FK_SENS_ETA_U, b in a:16
        push!(pairs, (a, b))               # eta--eta
    end
    for a in FK_SENS_ETA_U, b in 1:9
        push!(pairs, (a, b))               # eta--theta
    end
    pairs
end

@inline fk_sens_zindex(a::Int) = a <= 9 ? a : P + (a - 9)

function fk_sens_rhs(x::Vector{Float64}, S::Matrix{Float64}, T::Array{Float64,3},
                     u::Vector{Float64})
    dx, fx, fu, fxx, fxu, fuu = fk_rhs_sensitivity_derivatives(x, u)
    dS = fx * S + fu
    dT = zeros(8, FK_SENS_U, FK_SENS_U)
    for (a, b) in FK_SENS_PAIRS
        for l in 1:8
            value = dot(view(fx, l, :), view(T, :, a, b)) + fuu[l, a, b]
            for j in 1:8
                value += fxu[l, j, b] * S[j, a] + fxu[l, j, a] * S[j, b]
                for k in 1:8
                    value += fxx[l, j, k] * S[j, a] * S[k, b]
                end
            end
            dT[l, a, b] = value
            dT[l, b, a] = value
        end
    end
    return dx, dS, dT
end

function fk_sens_rk4_step(x::Vector{Float64}, S::Matrix{Float64}, T::Array{Float64,3},
                          h::Float64, u::Vector{Float64})
    k1x, k1s, k1t = fk_sens_rhs(x, S, T, u)
    k2x, k2s, k2t = fk_sens_rhs(x .+ (h / 2) .* k1x, S .+ (h / 2) .* k1s,
                                 T .+ (h / 2) .* k1t, u)
    k3x, k3s, k3t = fk_sens_rhs(x .+ (h / 2) .* k2x, S .+ (h / 2) .* k2s,
                                 T .+ (h / 2) .* k2t, u)
    k4x, k4s, k4t = fk_sens_rhs(x .+ h .* k3x, S .+ h .* k3s, T .+ h .* k3t, u)
    return x .+ (h / 6) .* (k1x .+ 2 .* k2x .+ 2 .* k3x .+ k4x),
           S .+ (h / 6) .* (k1s .+ 2 .* k2s .+ 2 .* k3s .+ k4s),
           T .+ (h / 6) .* (k1t .+ 2 .* k2t .+ 2 .* k3t .+ k4t)
end

function fk_sens_structural_vector(theta::Vector{Float64}, eta::Vector{Float64})
    length(theta) == P || error("FK population vector has wrong length")
    length(eta) == Q || error("FK EBE vector has wrong length")
    return vcat(theta[1:9], eta)
end

"""Return prediction values, first derivatives, and required second derivatives.

The derivative coordinates are [theta_1:P; eta_1:Q].  State second
sensitivities are propagated only for eta--eta and eta--theta pairs, which are
the pairs used by the FOCEI Hessian, mixed score derivative, and dG/dz.
"""
function fk_sens_predict(subj::FKSubject, theta::Vector{Float64}, eta::Vector{Float64})
    u = fk_sens_structural_vector(theta, eta)
    x = zeros(8)
    x[1] = subj.dose
    S = zeros(8, FK_SENS_U)
    T = zeros(8, FK_SENS_U, FK_SENS_U)
    mu = zeros(length(subj.times))
    dmu = [zeros(FK_SENS_D) for _ in subj.times]
    d2mu = [zeros(FK_SENS_D, FK_SENS_D) for _ in subj.times]
    tnow = 0.0
    v1 = exp(u[3] + u[12])
    circ0 = exp(u[7] + u[15])
    for index in subj.order
        target = subj.times[index]
        interval = target - tnow
        nstep = max(1, ceil(Int, interval / DT_MAX))
        h = interval / nstep
        for _ in 1:nstep
            x, S, T = fk_sens_rk4_step(x, S, T, h, u)
        end
        tnow = target
        grad_u = zeros(FK_SENS_U)
        hess_u = zeros(FK_SENS_U, FK_SENS_U)
        if subj.endpoint[index] == 1
            # mu = central / V1; log(V1) is u3 + u12.
            mu[index] = x[2] / v1
            for a in 1:FK_SENS_U
                la = (a == 3 || a == 12) ? 1.0 : 0.0
                grad_u[a] = S[2, a] / v1 - mu[index] * la
                for b in 1:FK_SENS_U
                    lb = (b == 3 || b == 12) ? 1.0 : 0.0
                    hess_u[a, b] = T[2, a, b] / v1 - S[2, a] / v1 * lb -
                                    S[2, b] / v1 * la + mu[index] * la * lb
                end
            end
        else
            # mu = circulating state + Circ0; log(Circ0) is u7 + u15.
            mu[index] = x[8] + circ0
            for a in 1:FK_SENS_U
                la = (a == 7 || a == 15) ? 1.0 : 0.0
                grad_u[a] = S[8, a] + circ0 * la
                for b in 1:FK_SENS_U
                    lb = (b == 7 || b == 15) ? 1.0 : 0.0
                    hess_u[a, b] = T[8, a, b] + circ0 * la * lb
                end
            end
        end
        for a in 1:FK_SENS_U
            za = fk_sens_zindex(a)
            dmu[index][za] = grad_u[a]
            for b in 1:FK_SENS_U
                d2mu[index][za, fk_sens_zindex(b)] = hess_u[a, b]
            end
        end
    end
    return mu, dmu, d2mu
end

@inline function fk_sens_residual_indices(endpoint::Int)
    return endpoint == 1 ? (17, 18) : (19, 20)
end

function fk_sens_variance_derivatives(mu::Float64, dmu::Vector{Float64}, d2mu::Matrix{Float64},
                                      theta::Vector{Float64}, endpoint::Int)
    add_idx, prop_idx = fk_sens_residual_indices(endpoint)
    add2 = exp(2 * theta[add_idx])
    prop2 = exp(2 * theta[prop_idx])
    v = add2 + prop2 * mu^2
    dv = zeros(FK_SENS_D)
    d2v = zeros(FK_SENS_D, FK_SENS_D)
    for a in 1:FK_SENS_D
        da = a == add_idx ? 1.0 : 0.0
        pa = a == prop_idx ? 1.0 : 0.0
        dv[a] = 2 * add2 * da + 2 * prop2 * pa * mu^2 + 2 * prop2 * mu * dmu[a]
        for b in 1:FK_SENS_D
            db = b == add_idx ? 1.0 : 0.0
            pb = b == prop_idx ? 1.0 : 0.0
            d2v[a, b] = 4 * add2 * da * db + 4 * prop2 * pa * pb * mu^2 +
                         4 * prop2 * mu * (pa * dmu[b] + pb * dmu[a]) +
                         2 * prop2 * (dmu[a] * dmu[b] + mu * d2mu[a, b])
        end
    end
    return v, dv, d2v, prop2
end

function fk_sens_add_observation!(h::Base.RefValue{Float64}, grad::Vector{Float64},
                                  hess::Matrix{Float64}, y::Float64, mu::Float64,
                                  dmu::Vector{Float64}, d2mu::Matrix{Float64},
                                  theta::Vector{Float64}, endpoint::Int)
    v, dv, d2v, _ = fk_sens_variance_derivatives(mu, dmu, d2mu, theta, endpoint)
    r = y - mu
    h[] += 0.5 * (log(2 * pi * v) + r^2 / v)
    phi_v = 0.5 / v - 0.5 * r^2 / v^2
    phi_r = r / v
    phi_vv = -0.5 / v^2 + r^2 / v^3
    phi_vr = -r / v^2
    phi_rr = 1 / v
    for a in 1:FK_SENS_D
        ra = -dmu[a]
        grad[a] += phi_v * dv[a] + phi_r * ra
        for b in 1:FK_SENS_D
            rb = -dmu[b]
            rab = -d2mu[a, b]
            hess[a, b] += phi_vv * dv[a] * dv[b] + phi_vr * (dv[a] * rb + ra * dv[b]) +
                          phi_rr * ra * rb + phi_v * d2v[a, b] + phi_r * rab
        end
    end
    return nothing
end

function fk_sens_add_prior!(h::Base.RefValue{Float64}, grad::Vector{Float64},
                            hess::Matrix{Float64}, theta::Vector{Float64}, eta::Vector{Float64})
    for j in 1:Q
        t = 9 + j
        e = P + j
        omega = exp(theta[t])
        invomega2 = inv(omega^2)
        h[] += log(omega) + 0.5 * eta[j]^2 * invomega2
        grad[t] += 1 - eta[j]^2 * invomega2
        grad[e] += eta[j] * invomega2
        hess[t, t] += 2 * eta[j]^2 * invomega2
        hess[t, e] += -2 * eta[j] * invomega2
        hess[e, t] += -2 * eta[j] * invomega2
        hess[e, e] += invomega2
    end
    return nothing
end

function fk_sens_focei_curvature_and_logdet_derivative(subj::FKSubject, theta::Vector{Float64},
                                                        mu, dmu, d2mu)
    G = zeros(Q, Q)
    dG = [zeros(Q, Q) for _ in 1:FK_SENS_D]
    for i in eachindex(mu)
        v, dv, _, prop2 = fk_sens_variance_derivatives(mu[i], dmu[i], d2mu[i], theta, subj.endpoint[i])
        jeta = view(dmu[i], (P + 1):(P + Q))
        dV = 2 .* prop2 .* mu[i] .* jeta
        for a in 1:Q, b in 1:Q
            G[a, b] += jeta[a] * jeta[b] / v + 0.5 * dV[a] * dV[b] / v^2
        end
        for z in 1:FK_SENS_D
            jz = dmu[i][z]
            for a in 1:Q, b in 1:Q
                jaz = d2mu[i][P + a, z]
                jbz = d2mu[i][P + b, z]
                dVaz = 2 * prop2 * (jz * jeta[a] + mu[i] * jaz) +
                       (z == fk_sens_residual_indices(subj.endpoint[i])[2] ? 4 * prop2 * mu[i] * jeta[a] : 0.0)
                dVbz = 2 * prop2 * (jz * jeta[b] + mu[i] * jbz) +
                       (z == fk_sens_residual_indices(subj.endpoint[i])[2] ? 4 * prop2 * mu[i] * jeta[b] : 0.0)
                dG[z][a, b] += -dv[z] * jeta[a] * jeta[b] / v^2 +
                                 (jaz * jeta[b] + jeta[a] * jbz) / v -
                                 dV[a] * dV[b] * dv[z] / v^3 +
                                 0.5 * (dVaz * dV[b] + dV[a] * dVbz) / v^2
            end
        end
    end
    for j in 1:Q
        omega_idx = 9 + j
        invomega2 = exp(-2 * theta[omega_idx])
        G[j, j] += invomega2
        dG[omega_idx][j, j] += -2 * invomega2
    end
    F = cholesky(Symmetric((G + transpose(G)) / 2); check=true)
    logdetG = 2 * sum(log, diag(F.L))
    # inv(Cholesky(G)) is already G^{-1}; a second transpose product would be G^{-2}.
    Ginv = inv(F)
    dld = zeros(FK_SENS_D)
    for z in 1:FK_SENS_D
        dld[z] = sum(Ginv .* dG[z])
    end
    return G, logdetG, dld
end

function fk_sensode_components(subj::FKSubject, theta::Vector{Float64}, eta::Vector{Float64};
                               need_logdet::Bool=false)
    mu, dmu, d2mu = fk_sens_predict(subj, theta, eta)
    h = Ref(0.0)
    grad = zeros(FK_SENS_D)
    hess = zeros(FK_SENS_D, FK_SENS_D)
    for i in eachindex(mu)
        fk_sens_add_observation!(h, grad, hess, subj.y[i], mu[i], dmu[i], d2mu[i], theta, subj.endpoint[i])
    end
    fk_sens_add_prior!(h, grad, hess, theta, eta)
    G, logdetG, dld = fk_sens_focei_curvature_and_logdet_derivative(subj, theta, mu, dmu, d2mu)
    return h[], grad, hess, G, logdetG, dld
end

# Inner EBE optimization only requires the q first-order eta sensitivities.
# This is the n(1+q) system in Almquist et al.; the mixed and second-order
# sensitivity states are reserved for the outer FOCEI derivative.
function fk_sens_first_rhs(x::Vector{Float64}, S::Matrix{Float64}, u::Vector{Float64})
    dx, fx, fu, _, _, _ = fk_rhs_sensitivity_derivatives(x, u)
    return dx, fx * S + view(fu, :, 10:16)
end

function fk_sens_first_rk4_step(x::Vector{Float64}, S::Matrix{Float64}, h::Float64, u::Vector{Float64})
    k1x, k1s = fk_sens_first_rhs(x, S, u)
    k2x, k2s = fk_sens_first_rhs(x .+ (h / 2) .* k1x, S .+ (h / 2) .* k1s, u)
    k3x, k3s = fk_sens_first_rhs(x .+ (h / 2) .* k2x, S .+ (h / 2) .* k2s, u)
    k4x, k4s = fk_sens_first_rhs(x .+ h .* k3x, S .+ h .* k3s, u)
    return x .+ (h / 6) .* (k1x .+ 2 .* k2x .+ 2 .* k3x .+ k4x),
           S .+ (h / 6) .* (k1s .+ 2 .* k2s .+ 2 .* k3s .+ k4s)
end

function fk_sens_first_predict(subj::FKSubject, theta::Vector{Float64}, eta::Vector{Float64})
    u = fk_sens_structural_vector(theta, eta)
    x = zeros(8)
    x[1] = subj.dose
    S = zeros(8, Q)
    mu = zeros(length(subj.times))
    J = zeros(length(subj.times), Q)
    tnow = 0.0
    v1 = exp(u[3] + u[12])
    circ0 = exp(u[7] + u[15])
    for index in subj.order
        target = subj.times[index]
        interval = target - tnow
        nstep = max(1, ceil(Int, interval / DT_MAX))
        h = interval / nstep
        for _ in 1:nstep
            x, S = fk_sens_first_rk4_step(x, S, h, u)
        end
        tnow = target
        if subj.endpoint[index] == 1
            mu[index] = x[2] / v1
            for a in 1:Q
                J[index, a] = S[2, a] / v1 - (a == 3 ? mu[index] : 0.0)
            end
        else
            mu[index] = x[8] + circ0
            for a in 1:Q
                J[index, a] = S[8, a] + (a == 6 ? circ0 : 0.0)
            end
        end
    end
    return mu, J
end

function fk_sensode_inner_value_score_curvature(subj::FKSubject, theta::Vector{Float64}, eta::Vector{Float64})
    mu, J = fk_sens_first_predict(subj, theta, eta)
    h = 0.0
    g = zeros(Q)
    G = zeros(Q, Q)
    for i in eachindex(mu)
        add_idx, prop_idx = fk_sens_residual_indices(subj.endpoint[i])
        add2 = exp(2 * theta[add_idx])
        prop2 = exp(2 * theta[prop_idx])
        v = add2 + prop2 * mu[i]^2
        r = subj.y[i] - mu[i]
        h += 0.5 * (log(2 * pi * v) + r^2 / v)
        phi_v = 0.5 / v - 0.5 * r^2 / v^2
        phi_r = r / v
        phi_mu = 2 * prop2 * mu[i] * phi_v - phi_r
        for a in 1:Q
            g[a] += phi_mu * J[i, a]
        end
        dV = 2 .* prop2 .* mu[i] .* view(J, i, :)
        for a in 1:Q, b in 1:Q
            G[a, b] += J[i, a] * J[i, b] / v + 0.5 * dV[a] * dV[b] / v^2
        end
    end
    for j in 1:Q
        omega = exp(theta[9 + j])
        invomega2 = inv(omega^2)
        h += log(omega) + 0.5 * eta[j]^2 * invomega2
        g[j] += eta[j] * invomega2
        G[j, j] += invomega2
    end
    return h, g, G
end
function fk_sensode_solve_mode(subj::FKSubject, theta::Vector{Float64};
                                maxiter::Int=parse(Int, get(ENV, "FK_MAXITER_ETA", "50")),
                                tol::Float64=parse(Float64, get(ENV, "FK_EBE_TOL", "5e-6")))
    eta = zeros(Q)
    for _ in 1:maxiter
        value, g, G = fk_sensode_inner_value_score_curvature(subj, theta, eta)
        norm(g) < tol && break
        step = newton_step(G, g)
        isfinite(norm(step)) || break
        accepted = false
        alpha = 1.0
        for _ in 1:16
            trial = eta .- alpha .* step
            # The published primal RHS floors circulating cells only while
            # exploring deliberately perturbed trial points. The analytic
            # sensitivity equations are valid on the strictly positive branch,
            # so reject a trial that leaves that branch rather than allowing a
            # fractional power to abort a threaded EBE solve.
            trial_value = try
                fk_sensode_inner_value_score_curvature(subj, theta, trial)[1]
            catch err
                err isa DomainError ? Inf : rethrow()
            end
            if isfinite(trial_value) && trial_value <= value - 1.0e-4 * alpha * dot(g, step)
                eta = trial
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    final_grad = fk_sensode_inner_value_score_curvature(subj, theta, eta)[2]
    return eta, norm(final_grad)
end

function fk_sensode_solve_all_modes(subjects, theta)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    norms = zeros(length(subjects))
    maxiter = parse(Int, get(ENV, "FK_MAXITER_ETA", "50"))
    tol = parse(Float64, get(ENV, "FK_EBE_TOL", "5e-6"))
    @threads for i in eachindex(subjects)
        etas[i], norms[i] = fk_sensode_solve_mode(subjects[i], theta; maxiter=maxiter, tol=tol)
    end
    return etas, maximum(norms), count(<(tol), norms)
end

function fk_sensode_mode_value(subjects, theta::Vector{Float64})
    etas, max_score, nconv = fk_sensode_solve_all_modes(subjects, theta)
    values = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        h, _, G = fk_sensode_inner_value_score_curvature(subjects[i], theta, etas[i])
        F = cholesky(Symmetric((G + transpose(G)) / 2); check=true)
        values[i] = 2 * (h + sum(log, diag(F.L)))
    end
    return sum(values), max_score, nconv, etas
end
function fk_almquist_sensitivity_ode_subject_value_grad(subj::FKSubject, theta::Vector{Float64}, eta::Vector{Float64})
    h, grad, hess, _, logdetG, dld = fk_sensode_components(subj, theta, eta; need_logdet=true)
    eta_range = (P + 1):(P + Q)
    H = hess[eta_range, eta_range]
    B = hess[eta_range, 1:P]
    S = -exact_hessian_solve(H, B)
    value = 2 * (h + 0.5 * logdetG)
    outer_grad = 2 .* (grad[1:P] .+ 0.5 .* dld[1:P] .+ transpose(S) * (0.5 .* dld[eta_range]))
    return value, Vector{Float64}(outer_grad)
end

function fk_almquist_sensitivity_ode_value_grad(subjects, theta)
    etas, maxscore, nconv = fk_sensode_solve_all_modes(subjects, theta)
    values = zeros(length(subjects))
    gradients = [zeros(P) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = fk_almquist_sensitivity_ode_subject_value_grad(subjects[i], theta, etas[i])
    end
    return sum(values), vec(sum(reduce(hcat, gradients), dims=2)), maxscore, nconv
end
