#!/usr/bin/env julia

# Reproducible scale benchmark for the FOCEI derivative organizations.
#
# The structural model is the eight-state Friberg--Karlsson PK--
# myelosuppression model used in the Torsten population-model example:
# https://metrumresearchgroup.github.io/Torsten/example/pkpd-pop/
#
# This script keeps the published structural equations, while the default
# simulation design is deliberately more informative than the 15-subject
# tutorial data set: 51 subjects are distributed equally across 50-, 80-, and
# 110-mg oral doses (stored as micrograms) and ANC is sampled daily through
# the decline/nadir/recovery window.
# A named legacy design preserves the original tutorial-like configuration.
# Both designs use combined additive-plus-proportional residual error for PK
# and ANC, which makes the conditional covariance EBE-dependent and therefore
# creates a genuine FOCEI (rather than FOCE) benchmark.
#
# Dimensions in this implementation:
#   * 8 ODE states (depot, central, peripheral, and five ANC states)
#   * 9 structural population parameters
#   * 7 diagonal IIV parameters / EBEs
#   * 4 residual-error parameters (additive + proportional for each endpoint)
#   * 20 estimated population parameters overall.
#
# The comparison uses exactly the same simulated data, theta, zero EBE starts,
# inner Newton solver, tolerances, and Julia thread count for both methods.
# It reports both (i) the derivative-organizing cost conditional on the same
# converged EBEs and (ii) the end-to-end gradient-evaluation cost including
# re-solving all EBEs.  The latter is the operationally relevant number.

using ForwardDiff
using LinearAlgebra
using Random
using Statistics
using Base.Threads

const Q = 7
const P = 20
const DT_MAX = 0.50

# FK_DESIGN=legacy_15 reproduces the original single-dose, 15-subject
# tutorial-like schedule. Amounts are stored in micrograms, so the historic
# value 80_000 represents an 80-mg dose. The enriched design is the default
# for all new simulation work: 51 subjects give exactly 17 subjects at each
# dose level. The dense design retains those subjects and dose arms, while
# doubling the per-subject PK--PD sampling burden. The targeted design adds
# 6-hour ANC observations across the predicted decline--nadir--recovery window.
const FK_DESIGN = get(ENV, "FK_DESIGN", "enriched_51_subjects")
const FK_N_SUBJECTS_DEFAULT = FK_DESIGN == "legacy_15" ? 15 :
                              FK_DESIGN == "enriched_51_subjects" ? 51 :
                              FK_DESIGN == "dense_pk_pd_51subjects" ? 51 :
                              FK_DESIGN == "targeted_pkpd_51subjects" ? 51 :
                              error("unknown FK_DESIGN=$FK_DESIGN; use legacy_15, enriched_51_subjects, dense_pk_pd_51subjects, or targeted_pkpd_51subjects")
const PK_TIMES = FK_DESIGN in ("dense_pk_pd_51subjects", "targeted_pkpd_51subjects") ?
    Float64[0.083, 0.167, 0.25, 0.333, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6,
            8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168] :
    Float64[0.083, 0.167, 0.25, 0.5, 0.75, 1, 2, 3, 4, 6, 8, 12, 18, 24]
const ANC_TIMES = FK_DESIGN == "legacy_15" ?
    Float64[0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672] :
    FK_DESIGN == "targeted_pkpd_51subjects" ?
    sort(unique(vcat(collect(0.0:12.0:672.0), collect(96.0:6.0:336.0)))) :
    FK_DESIGN == "dense_pk_pd_51subjects" ?
    collect(0.0:12.0:672.0) :
    # Baseline, day 2, daily during days 3--21, then every 48 h to day 28.
    vcat(Float64[0, 48], collect(72.0:24.0:504.0), collect(528.0:48.0:672.0))
const FK_DOSE_LEVELS_MG = FK_DESIGN == "legacy_15" ? Float64[80.0] :
                          Float64[50.0, 80.0, 110.0]
const FK_DOSE_LEVELS = 1_000.0 .* FK_DOSE_LEVELS_MG # model amount unit: ug

struct FKSubject
    id::Int
    dose::Float64
    times::Vector{Float64}
    endpoint::Vector{Int} # 1 = PK concentration, 2 = ANC
    y::Vector{Float64}
    order::Vector{Int}
end

"""The published Friberg--Karlsson parameterization, plus combined error."""
function true_theta()
    # Population parameters: CL, Q, V1, V2, ka, MTT, Circ0, gamma, alpha.
    structural = log.([9.54, 15.40, 37.40, 101.70, 2.00, 113.70, 4.76, 0.171, 2.20e-4])
    # IIV: CL, Q, V1, V2, MTT, Circ0, alpha, using the reported scale.
    omega = log.([0.223, 0.339, 0.264, 0.257, 0.177, 0.188, 0.409])
    # Combined residual errors: PK additive, PK proportional, ANC additive,
    # ANC proportional. The proportional components track the source's
    # approximately 10% residual scale; the additive components are modest.
    residual = log.([5.0, 0.10, 0.05, 0.10])
    theta = vcat(structural, omega, residual)
    length(theta) == P || error("incorrect population dimension")
    return theta
end

function individual_parameters(theta, eta)
    return (
        cl = exp(theta[1] + eta[1]),
        q = exp(theta[2] + eta[2]),
        v1 = exp(theta[3] + eta[3]),
        v2 = exp(theta[4] + eta[4]),
        ka = exp(theta[5]),
        mtt = exp(theta[6] + eta[5]),
        circ0 = exp(theta[7] + eta[6]),
        gamma = exp(theta[8]),
        alpha = exp(theta[9] + eta[7]),
    )
end

function ode_rhs(x, pars)
    T = eltype(x)
    dx = Vector{T}(undef, 8)
    k10 = pars.cl / pars.v1
    k12 = pars.q / pars.v1
    k21 = pars.q / pars.v2
    ktr = 4 / pars.mtt
    conc = x[2] / pars.v1
    edrug = pars.alpha * conc
    prol = x[4] + pars.circ0
    tr1 = x[5] + pars.circ0
    tr2 = x[6] + pars.circ0
    tr3 = x[7] + pars.circ0
    # The mode evaluations stay strictly positive. The floor just prevents an
    # invalid fractional power while testing deliberately perturbed starts.
    circ = max(x[8] + pars.circ0, convert(T, 1.0e-8))
    dx[1] = -pars.ka * x[1]
    dx[2] = pars.ka * x[1] - (k10 + k12) * x[2] + k21 * x[3]
    dx[3] = k12 * x[2] - k21 * x[3]
    dx[4] = ktr * prol * ((1 - edrug) * (pars.circ0 / circ)^pars.gamma - 1)
    dx[5] = ktr * (prol - tr1)
    dx[6] = ktr * (tr1 - tr2)
    dx[7] = ktr * (tr2 - tr3)
    dx[8] = ktr * (tr3 - circ)
    return dx
end

function rk4_step(x, h, pars)
    k1 = ode_rhs(x, pars)
    k2 = ode_rhs(x .+ (h / 2) .* k1, pars)
    k3 = ode_rhs(x .+ (h / 2) .* k2, pars)
    k4 = ode_rhs(x .+ h .* k3, pars)
    return x .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function predict_subject(subj::FKSubject, theta, eta)
    pars = individual_parameters(theta, eta)
    T = typeof(theta[1] + eta[1])
    x = zeros(T, 8)
    x[1] = convert(T, subj.dose)
    pred = Vector{T}(undef, length(subj.times))
    tnow = 0.0
    for index in subj.order
        target = subj.times[index]
        interval = target - tnow
        nstep = max(1, ceil(Int, interval / DT_MAX))
        h = interval / nstep
        for _ in 1:nstep
            x = rk4_step(x, h, pars)
        end
        tnow = target
        pred[index] = subj.endpoint[index] == 1 ? x[2] / pars.v1 : x[8] + pars.circ0
    end
    return pred
end

variance(mu, sigma_add, sigma_prop) = sigma_add^2 + (sigma_prop * mu)^2

function h_i(subj::FKSubject, theta, eta)
    pred = predict_subject(subj, theta, eta)
    sigma_add_pk, sigma_prop_pk = exp(theta[17]), exp(theta[18])
    sigma_add_anc, sigma_prop_anc = exp(theta[19]), exp(theta[20])
    z = zero(theta[1] + eta[1])
    for j in eachindex(pred)
        sigma_add, sigma_prop = subj.endpoint[j] == 1 ?
            (sigma_add_pk, sigma_prop_pk) : (sigma_add_anc, sigma_prop_anc)
        v = variance(pred[j], sigma_add, sigma_prop)
        r = subj.y[j] - pred[j]
        z += 0.5 * (log(2 * pi * v) + r^2 / v)
    end
    omega = exp.(theta[10:16])
    for j in 1:Q
        z += log(omega[j]) + 0.5 * (eta[j] / omega[j])^2
    end
    return z
end

function focei_curvature(subj::FKSubject, theta, eta)
    pred = predict_subject(subj, theta, eta)
    J = ForwardDiff.jacobian(e -> predict_subject(subj, theta, e), eta)
    sigma_add_pk, sigma_prop_pk = exp(theta[17]), exp(theta[18])
    sigma_add_anc, sigma_prop_anc = exp(theta[19]), exp(theta[20])
    T = eltype(J)
    v = Vector{T}(undef, length(pred))
    dV = Matrix{T}(undef, length(pred), Q)
    for i in eachindex(pred)
        sigma_add, sigma_prop = subj.endpoint[i] == 1 ?
            (sigma_add_pk, sigma_prop_pk) : (sigma_add_anc, sigma_prop_anc)
        v[i] = variance(pred[i], sigma_add, sigma_prop)
        @inbounds for j in 1:Q
            dV[i, j] = 2 * sigma_prop^2 * pred[i] * J[i, j]
        end
    end
    G = transpose(J) * Diagonal(inv.(v)) * J +
        0.5 .* transpose(dV) * Diagonal(inv.(v .* v)) * dV
    omega = exp.(theta[10:16])
    for j in 1:Q
        G[j, j] += one(T) / omega[j]^2
    end
    return G
end

function logdet_spd(A)
    F = cholesky(Symmetric((A + transpose(A)) / 2); check=true)
    return 2 * sum(log, diag(F.L))
end

function focei_subject_fixed_eta(subj::FKSubject, theta, eta)
    return 2 * (h_i(subj, theta, eta) + 0.5 * logdet_spd(focei_curvature(subj, theta, eta)))
end

function newton_step(H, g)
    Hs = Matrix{Float64}((H + transpose(H)) / 2)
    # Small diagonal regularization is only a numerical guard for exploratory
    # starts; it was never selected for the retained converged modes below.
    for jitter in (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
        F = cholesky(Symmetric(Hs + jitter * I); check=false)
        if issuccess(F)
            return F \ g
        end
    end
    return Hs \ g
end

# At a converged conditional mode the exact Hessian is positive definite and
# the Cholesky path is used. Deliberately capped EBE calculations can be
# evaluated before that condition holds; their exact Hessian is nevertheless
# the matrix required by the sensitivity formula, so use a pivoted symmetric-
# indefinite factorization rather than treating non-positive definiteness as
# a numerical failure.
function exact_hessian_solve(H, rhs)
    Hs = Matrix{Float64}((H + transpose(H)) / 2)
    chol = cholesky(Symmetric(Hs); check=false)
    if issuccess(chol)
        return chol \ rhs
    end
    fac = bunchkaufman(Symmetric(Hs); check=false)
    if issuccess(fac)
        solution = fac \ rhs
        all(isfinite, solution) && return solution
    end
    return Hs \ rhs
end

function solve_mode(subj::FKSubject, theta; maxiter::Int=50, tol::Float64=parse(Float64, get(ENV, "FK_EBE_TOL", "5e-6")))
    eta = zeros(Q)
    f = e -> h_i(subj, theta, e)
    for _ in 1:maxiter
        f0 = Float64(f(eta))
        g = Vector{Float64}(ForwardDiff.gradient(f, eta))
        norm(g) < tol && break
        # As in a standard FOCEI EBE calculation, use the positive-definite
        # expected working curvature for the line-searched mode iterations;
        # the score itself remains the exact conditional score. This is much
        # more robust than an unrestricted exact-Hessian Newton step in the
        # delayed myelosuppression portion of the model.
        G = focei_curvature(subj, theta, eta)
        step = newton_step(G, g)
        isfinite(norm(step)) || break
        accepted = false
        alpha = 1.0
        for _ in 1:16
            trial = eta .- alpha .* step
            ftrial = Float64(f(trial))
            if isfinite(ftrial) && ftrial <= f0 - 1.0e-4 * alpha * dot(g, step)
                eta = trial
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    score = Vector{Float64}(ForwardDiff.gradient(f, eta))
    return eta, norm(score)
end

function solve_all_modes(subjects, theta;
                         maxiter::Int=parse(Int, get(ENV, "FK_MAXITER_ETA", "50")),
                         tol::Float64=parse(Float64, get(ENV, "FK_EBE_TOL", "5e-6")))
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    norms = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        etas[i], norms[i] = solve_mode(subjects[i], theta; maxiter=maxiter, tol=tol)
    end
    return etas, maximum(norms), count(<(tol), norms)
end

# Almquist et al.'s forward-sensitivity organization: explicitly materialize
# B = d g / d theta (q x p), then S = -H^{-1} B.
function almquist_subject_value_grad(subj::FKSubject, theta, eta)
    H = ForwardDiff.hessian(e -> h_i(subj, theta, e), eta)
    value = focei_subject_fixed_eta(subj, theta, eta)
    dh = ForwardDiff.gradient(x -> h_i(subj, x, eta), theta)
    dldtheta = ForwardDiff.gradient(x -> logdet_spd(focei_curvature(subj, x, eta)), theta)
    dldeta = ForwardDiff.gradient(e -> logdet_spd(focei_curvature(subj, theta, e)), eta)
    B = ForwardDiff.jacobian(x -> ForwardDiff.gradient(e -> h_i(subj, x, e), eta), theta)
    Hs = Matrix{Float64}((H + transpose(H)) / 2)
    S = -exact_hessian_solve(Hs, Matrix{Float64}(B))
    grad = 2 .* (dh .+ 0.5 .* dldtheta .+ transpose(S) * (0.5 .* dldeta))
    return Float64(value), Vector{Float64}(grad)
end

# Directional-JVP organization: solve the adjoint H lambda = .5 d log|G|/deta
# and evaluate B' lambda as the theta gradient of one eta-directional JVP.
function directional_score_contraction(subj::FKSubject, theta, eta, lambda)
    return ForwardDiff.derivative(t -> h_i(subj, theta, eta .+ t .* lambda), 0.0)
end

function directional_jvp_subject_value_grad(subj::FKSubject, theta, eta)
    H = ForwardDiff.hessian(e -> h_i(subj, theta, e), eta)
    value = focei_subject_fixed_eta(subj, theta, eta)
    dh = ForwardDiff.gradient(x -> h_i(subj, x, eta), theta)
    dldtheta = ForwardDiff.gradient(x -> logdet_spd(focei_curvature(subj, x, eta)), theta)
    dldeta = ForwardDiff.gradient(e -> logdet_spd(focei_curvature(subj, theta, e)), eta)
    Hs = Matrix{Float64}((H + transpose(H)) / 2)
    lambda = exact_hessian_solve(Hs, 0.5 .* Vector{Float64}(dldeta))
    contraction = ForwardDiff.gradient(x -> directional_score_contraction(subj, x, eta, lambda), theta)
    grad = 2 .* (dh .+ 0.5 .* dldtheta .- contraction)
    return Float64(value), Vector{Float64}(grad)
end

function gradient_from_modes(method, subjects, theta, etas)
    values = zeros(length(subjects))
    gradients = [zeros(P) for _ in subjects]
    @threads for i in eachindex(subjects)
        if method == :almquist
            values[i], gradients[i] = almquist_subject_value_grad(subjects[i], theta, etas[i])
        elseif method == :directional_jvp
            values[i], gradients[i] = directional_jvp_subject_value_grad(subjects[i], theta, etas[i])
        else
            error("unknown method $method")
        end
    end
    return sum(values), vec(sum(reduce(hcat, gradients), dims=2))
end

function full_value_grad(method, subjects, theta)
    etas, maxscore, nconv = solve_all_modes(subjects, theta)
    value, grad = gradient_from_modes(method, subjects, theta, etas)
    return value, grad, maxscore, nconv
end

"""Assign dose amounts in ug cyclically, yielding equal group sizes when n is a multiple of three."""
subject_dose(id::Integer) = FK_DOSE_LEVELS[mod1(id, length(FK_DOSE_LEVELS))]

function simulate_subject(rng, id, theta)
    omega = exp.(theta[10:16])
    eta = omega .* randn(rng, Q)
    times = vcat(PK_TIMES, ANC_TIMES)
    endpoint = vcat(fill(1, length(PK_TIMES)), fill(2, length(ANC_TIMES)))
    order = sortperm(times)
    dose = subject_dose(id)
    blank = FKSubject(id, dose, times, endpoint, zeros(length(times)), order)
    mu = Vector{Float64}(predict_subject(blank, theta, eta))
    y = similar(mu)
    sigma_add_pk, sigma_prop_pk = exp(theta[17]), exp(theta[18])
    sigma_add_anc, sigma_prop_anc = exp(theta[19]), exp(theta[20])
    for j in eachindex(mu)
        sa, sp = endpoint[j] == 1 ? (sigma_add_pk, sigma_prop_pk) : (sigma_add_anc, sigma_prop_anc)
        y[j] = max(1.0e-8, mu[j] + sqrt(variance(mu[j], sa, sp)) * randn(rng))
    end
    return FKSubject(id, dose, times, endpoint, y, order)
end

"""Generate a deterministic Friberg--Karlsson population for the selected design."""
function simulate_population(rng, theta; nsubjects::Int=FK_N_SUBJECTS_DEFAULT)
    nsubjects > 0 || error("nsubjects must be positive")
    return [simulate_subject(rng, i, theta) for i in 1:nsubjects]
end

function median_timing(f, repeats)
    samples = Float64[]
    for _ in 1:repeats
        GC.gc()
        elapsed = @elapsed f()
        push!(samples, elapsed)
    end
    return samples, median(samples)
end

function main()
    theta = true_theta()
    rng = MersenneTwister(20260822)
    subjects = simulate_population(rng, theta)

    # Evaluate near, but not exactly at, the simulation truth so the EBE solve
    # and FOCEI correction remain nontrivial.  Both methods receive this exact
    # same vector and use cold zero EBE starts during the end-to-end timing.
    start = copy(theta)
    start[1:9] .+= [0.05, -0.04, 0.03, -0.03, 0.04, -0.04, 0.03, 0.02, -0.05]
    start[10:16] .+= 0.03
    start[17:20] .+= [-0.08, 0.04, -0.05, 0.03]

    println("Friberg--Karlsson FOCEI scale benchmark")
    println("threads=$(nthreads()) design=$(FK_DESIGN) subjects=$(length(subjects)) states=8 structural=9 iiv=$(Q) residual=4 population=$(P)")
    println("seed=20260822 dt_max=$(DT_MAX) doses_mg=$(join(FK_DOSE_LEVELS_MG, ';')) PK_observations=$(length(PK_TIMES)) ANC_observations=$(length(ANC_TIMES))")

    # Warm compilation once. These calls are deliberately discarded.
    etas, maxscore, nconv = solve_all_modes(subjects, start)
    println("mode_check=max_score=$(maxscore) converged=$(nconv)/$(length(subjects))")
    nconv == length(subjects) || error("all EBEs must converge before timing")
    value_a, grad_a = gradient_from_modes(:almquist, subjects, start, etas)
    value_d, grad_d = gradient_from_modes(:directional_jvp, subjects, start, etas)
    scale = max(1.0, norm(grad_a), norm(grad_d))
    println("validation=value_abs_difference=$(abs(value_a - value_d)) gradient_relative_difference=$(norm(grad_a - grad_d) / scale)")

    # Conditional timing isolates the q-by-p forward-sensitivity construction.
    cond_a, med_cond_a = median_timing(() -> gradient_from_modes(:almquist, subjects, start, etas), 3)
    cond_d, med_cond_d = median_timing(() -> gradient_from_modes(:directional_jvp, subjects, start, etas), 3)

    # End-to-end timing includes fresh, matched EBE solutions and is the main
    # reported comparison. A second call keeps wall-clock noise manageable.
    full_a, med_full_a = median_timing(() -> full_value_grad(:almquist, subjects, start), 2)
    full_d, med_full_d = median_timing(() -> full_value_grad(:directional_jvp, subjects, start), 2)

    println("conditional_seconds_almquist=$(join(cond_a, ';')) median=$(med_cond_a)")
    println("conditional_seconds_directional_jvp=$(join(cond_d, ';')) median=$(med_cond_d)")
    println("conditional_speedup_almquist_over_directional=$(med_cond_a / med_cond_d)")
    println("end_to_end_seconds_almquist=$(join(full_a, ';')) median=$(med_full_a)")
    println("end_to_end_seconds_directional_jvp=$(join(full_d, ';')) median=$(med_full_d)")
    println("end_to_end_speedup_almquist_over_directional=$(med_full_a / med_full_d)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
