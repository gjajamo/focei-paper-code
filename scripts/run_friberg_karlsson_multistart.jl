#!/usr/bin/env julia

# Matched-start optimization experiment for the Friberg--Karlsson FOCEI case
# study. Model equations and data simulation are defined in the validated
# scale-benchmark file; this runner supplies the population optimizers and
# records the four manuscript gradient strategies plus the Almquist comparator.

using Optim
using DelimitedFiles

include(joinpath(@__DIR__, "run_friberg_karlsson_scale_benchmark.jl"))
include(joinpath(@__DIR__, "..", "src", "almquist_fk_sensitivity_ode.jl"))

const FK_METHODS_DEFAULT = "FULL_IMPLICIT_DIRECTIONAL_JVP,ONE_STEP_NEWTON,STOP,FD,ALMQUIST_FORWARD"

function fk_bounds()
    lo_struct = log.([0.5, 1.0, 10.0, 20.0, 0.2, 40.0, 1.0, 0.05, 5.0e-5])
    hi_struct = log.([30.0, 50.0, 100.0, 300.0, 5.0, 250.0, 10.0, 1.0, 1.0e-3])
    lo_omega = fill(log(0.05), Q)
    hi_omega = fill(log(1.0), Q)
    lo_resid = log.([0.01, 0.01, 0.005, 0.01])
    hi_resid = log.([50.0, 0.5, 1.0, 0.5])
    return vcat(lo_struct, lo_omega, lo_resid), vcat(hi_struct, hi_omega, hi_resid)
end

function fk_sample_starts(theta, lo, hi, nstarts;
                          seed::Int=20260823,
                          log_sd::Float64=parse(Float64, get(ENV, "FK_START_LOG_SD", "0.30")))
    log_sd >= 0.0 || error("FK_START_LOG_SD must be nonnegative")
    rng = MersenneTwister(seed)
    starts = Matrix{Float64}(undef, nstarts, P)
    starts[1, :] .= theta
    for s in 2:nstarts
        starts[s, :] .= min.(max.(theta .+ log_sd .* randn(rng, P), lo), hi)
    end
    return starts
end

function fk_mode_value(subjects, theta)
    etas, max_score, nconv = solve_all_modes(subjects, theta)
    values = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        values[i] = Float64(focei_subject_fixed_eta(subjects[i], theta, etas[i]))
    end
    return sum(values), max_score, nconv, etas
end

function fk_stop_subject_value_grad(subj, theta, eta)
    f = x -> focei_subject_fixed_eta(subj, x, eta)
    return Float64(f(theta)), Vector{Float64}(ForwardDiff.gradient(f, theta))
end

function fk_ad_spd_solve(H, b)
    A = (H + transpose(H)) / 2
    return cholesky(Symmetric(A); check=true) \ b
end

# The EBE is first solved independently, then treated as a detached starting
# value for precisely one exact-Newton correction.  Directly nesting an outer
# 20-dimensional ForwardDiff gradient through a Hessian and a log-determinant
# requires fourth-order dual arithmetic for this eight-state model and is not
# a meaningful implementation of the one-step method.  Instead we evaluate
# the tangent of that single Newton map explicitly.  At a converged EBE this
# is exactly -H^{-1}B, the stationary one-step limit described in the paper.
#
# If a deliberately loose EBE tolerance is used, the optional residual term
# H^{-1}(dH/dtheta)H^{-1}g can be enabled for an exact finite-residual tangent.
# Production runs use a 5e-6 score tolerance, for which that O(||g||) term is
# negligible relative to the reported precision while avoiding third-order
# directional derivatives in every population-parameter direction.
function fk_hessian_bilinear(subj, theta, eta, left, right)
    f = e -> h_i(subj, theta, e)
    return ForwardDiff.derivative(
        s -> ForwardDiff.derivative(t -> f(eta .+ s .* left .+ t .* right), 0.0),
        0.0,
    )
end

function fk_one_step_subject_value_grad(subj, theta, eta_star;
                                        residual_correction::Bool=false)
    eta = Vector{Float64}(eta_star)
    g = Vector{Float64}(ForwardDiff.gradient(e -> h_i(subj, theta, e), eta))
    H = Matrix{Float64}(ForwardDiff.hessian(e -> h_i(subj, theta, e), eta))
    Hs = (H + transpose(H)) / 2
    u = exact_hessian_solve(Hs, g)
    eta_next = eta .- u

    value = Float64(focei_subject_fixed_eta(subj, theta, eta_next))
    direct = Vector{Float64}(ForwardDiff.gradient(x -> focei_subject_fixed_eta(subj, x, eta_next), theta))
    feta = Vector{Float64}(ForwardDiff.gradient(e -> focei_subject_fixed_eta(subj, theta, e), eta_next))
    B = Matrix{Float64}(ForwardDiff.jacobian(
        x -> ForwardDiff.gradient(e -> h_i(subj, x, e), eta), theta,
    ))
    sensitivity = -exact_hessian_solve(Hs, B)
    grad = direct .+ transpose(sensitivity) * feta

    if residual_correction && norm(g) > 0.0
        # w' (dH/dtheta) u, where w = H^{-T} dF/deta and u = H^{-1} g.
        # This is the O(||g||) finite-residual correction to the stationary
        # one-step sensitivity.
        w = exact_hessian_solve(Hs, feta)
        correction = ForwardDiff.gradient(
            x -> fk_hessian_bilinear(subj, x, eta, w, u), theta,
        )
        grad .+= Vector{Float64}(correction)
    end
    return value, Vector{Float64}(grad)
end

# Exact-Hessian Laplace criterion organized with the same directional
# contraction as the FOCEI implicit implementation. The EBE score equation
# and its Hessian are unchanged; only the determinant uses H rather than G.
function fk_laplace_subject_fixed_eta(subj, theta, eta)
    H = ForwardDiff.hessian(e -> h_i(subj, theta, e), eta)
    return 2.0 * (h_i(subj, theta, eta) + 0.5 * logdet_spd(H))
end

function fk_laplace_directional_subject_value_grad(subj, theta, eta)
    H = ForwardDiff.hessian(e -> h_i(subj, theta, e), eta)
    value = fk_laplace_subject_fixed_eta(subj, theta, eta)
    dh = ForwardDiff.gradient(x -> h_i(subj, x, eta), theta)
    dldtheta = ForwardDiff.gradient(
        x -> logdet_spd(ForwardDiff.hessian(e -> h_i(subj, x, e), eta)), theta,
    )
    dldeta = ForwardDiff.gradient(
        e -> logdet_spd(ForwardDiff.hessian(ee -> h_i(subj, theta, ee), e)), eta,
    )
    Hs = Matrix{Float64}((H + transpose(H)) / 2)
    lambda = cholesky(Symmetric(Hs); check=true) \ (0.5 .* Vector{Float64}(dldeta))
    contraction = ForwardDiff.gradient(
        x -> directional_score_contraction(subj, x, eta, lambda), theta,
    )
    gradient = 2.0 .* (dh .+ 0.5 .* dldtheta .- contraction)
    return Float64(value), Vector{Float64}(gradient)
end

function fk_laplace_mode_value(subjects, theta)
    etas, max_score, nconv = solve_all_modes(subjects, theta)
    values = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        values[i] = Float64(fk_laplace_subject_fixed_eta(subjects[i], theta, etas[i]))
    end
    return sum(values), max_score, nconv
end

# A non-positive-definite exact Hessian makes the Laplace determinant undefined
# at that endpoint.  It is an invalid candidate, not a reason to discard the
# complete matched-start batch.  The optimizer itself already applies this
# convention to invalid trial evaluations through fk_ensure!.
function fk_laplace_recomputed_or_invalid(subjects, theta)
    try
        return fk_laplace_mode_value(subjects, theta)[1], false
    catch err
        @warn "Laplace endpoint has an undefined exact-Hessian determinant; recording invalid endpoint" exception=typeof(err)
        return Inf, true
    end
end

function fk_gradient_from_modes(method::String, subjects, theta, etas)
    values = zeros(length(subjects))
    gradients = [zeros(P) for _ in subjects]
    @threads for i in eachindex(subjects)
        if method == "FULL_IMPLICIT_DIRECTIONAL_JVP"
            values[i], gradients[i] = directional_jvp_subject_value_grad(subjects[i], theta, etas[i])
        elseif method == "ALMQUIST_FORWARD"
            values[i], gradients[i] = almquist_subject_value_grad(subjects[i], theta, etas[i])
        elseif method == "ONE_STEP_NEWTON"
            residual = lowercase(get(ENV, "FK_ONE_STEP_RESIDUAL_CORRECTION", "false")) in
                       ("1", "true", "yes", "on")
            values[i], gradients[i] = fk_one_step_subject_value_grad(
                subjects[i], theta, etas[i]; residual_correction=residual,
            )
        elseif method == "STOP"
            values[i], gradients[i] = fk_stop_subject_value_grad(subjects[i], theta, etas[i])
        elseif method in ("LAPLACE_DIRECTIONAL_IMPLICIT", "LAPLACE_IMPLICIT")
            values[i], gradients[i] = fk_laplace_directional_subject_value_grad(subjects[i], theta, etas[i])
        else
            error("unknown AD method $method")
        end
    end
    return sum(values), vec(sum(reduce(hcat, gradients), dims=2))
end

function fk_fd_value_grad(subjects, theta; step::Float64=1.0e-4)
    lo, hi = fk_bounds()
    f0, max_score, nconv, _ = fk_mode_value(subjects, theta)
    grad = zeros(P)
    # Do not nest subject threading inside a parameter-threaded FD loop. Each
    # perturbation uses the same eight-thread EBE solver as all AD methods.
    for j in 1:P
        h = step * max(1.0, abs(theta[j]))
        xp, xm = copy(theta), copy(theta)
        xp[j] = min(hi[j], theta[j] + h)
        xm[j] = max(lo[j], theta[j] - h)
        fp, _, _, _ = fk_mode_value(subjects, xp)
        fm, _, _, _ = fk_mode_value(subjects, xm)
        grad[j] = (fp - fm) / (xp[j] - xm[j])
    end
    return f0, grad, max_score, nconv
end

function fk_value_grad(method::String, subjects, theta)
    if method == "FD"
        return fk_fd_value_grad(subjects, theta)
    elseif method == "ALMQUIST_SENSITIVITY_ODE"
        return fk_almquist_sensitivity_ode_value_grad(subjects, theta)
    end
    etas, max_score, nconv = solve_all_modes(subjects, theta)
    value, grad = fk_gradient_from_modes(method, subjects, theta, etas)
    return value, grad, max_score, nconv
end

mutable struct FKCache
    valid::Bool
    x::Vector{Float64}
    value::Float64
    grad::Vector{Float64}
    max_score::Float64
    nconv::Int
    evaluations::Int
    failed::Bool
end

FKCache() = FKCache(false, zeros(P), Inf, zeros(P), NaN, 0, 0, false)

function fk_penalty(theta, lo, hi)
    below = min.(theta .- lo, 0.0)
    above = max.(theta .- hi, 0.0)
    outside = any(below .< 0.0) || any(above .> 0.0)
    d = below .+ above
    return outside, 1.0e12 + 1.0e8 * sum(abs2, d), 2.0e8 .* d
end

function fk_ensure!(cache, method, subjects, x, lo, hi)
    theta = Vector{Float64}(x)
    if cache.valid && cache.x == theta
        return
    end
    outside, pen, pgrad = fk_penalty(theta, lo, hi)
    if outside
        cache.x, cache.value, cache.grad = theta, pen, pgrad
        cache.max_score, cache.nconv, cache.failed = NaN, 0, true
        cache.valid = true
        return
    end
    try
        value, grad, max_score, nconv = fk_value_grad(method, subjects, theta)
        if !isfinite(value) || any(!isfinite, grad)
            error("non-finite FOCEI objective or derivative")
        end
        cache.x, cache.value, cache.grad = theta, Float64(value), Vector{Float64}(grad)
        cache.max_score, cache.nconv = Float64(max_score), Int(nconv)
        cache.failed = false
    catch err
        @warn "Friberg evaluation failed; returning penalty" method exception=typeof(err)
        cache.x, cache.value, cache.grad = theta, 1.0e12, zeros(P)
        cache.max_score, cache.nconv, cache.failed = NaN, 0, true
    end
    cache.evaluations += 1
    cache.valid = true
end

function fk_optimize_one(method, subjects, theta0, lo, hi; maxiter::Int=35)
    cache = FKCache()
    f = x -> begin
        fk_ensure!(cache, method, subjects, x, lo, hi)
        cache.value
    end
    function g!(g, x)
        fk_ensure!(cache, method, subjects, x, lo, hi)
        g .= cache.grad
        return g
    end
    GC.gc()
    wall0, cpu0 = time_ns(), time()
    result = optimize(
        f, g!, theta0, LBFGS(),
        Optim.Options(iterations=maxiter, g_tol=1.0e-5, f_reltol=1.0e-8, show_trace=false),
    )
    wall = (time_ns() - wall0) / 1.0e9
    theta = min.(max.(Vector{Float64}(Optim.minimizer(result)), lo), hi)
    fk_ensure!(cache, method, subjects, theta, lo, hi)
    canonical, max_score, nconv, endpoint_failed = try
        method == "ALMQUIST_SENSITIVITY_ODE" ? fk_sensode_mode_value(subjects, theta) : fk_mode_value(subjects, theta)
    catch err
        @warn "Friberg endpoint evaluation failed" method exception=typeof(err)
        (Inf, Inf, 0, true)
    end
    recomputed_method_value, endpoint_invalid = method in ("LAPLACE_DIRECTIONAL_IMPLICIT", "LAPLACE_IMPLICIT") ?
        fk_laplace_recomputed_or_invalid(subjects, theta) : (canonical, false)
    return (
        method=method,
        theta=theta,
        iterations=Optim.iterations(result),
        optimizer_converged=Optim.converged(result),
        method_value=cache.value,
        canonical_value=canonical,
        recomputed_method_value=recomputed_method_value,
        wall_sec=wall,
        cpu_sec=time() - cpu0,
        evaluations=cache.evaluations,
        max_score=max_score,
        nconv=nconv,
        failed=cache.failed || endpoint_invalid || endpoint_failed,
    )
end

function fk_write_rows(path, rows)
    open(path, "w") do io
        names = ["logCL", "logQ", "logV1", "logV2", "logKA", "logMTT", "logCIRC0",
                 "logGAMMA", "logALPHA", "logOM_CL", "logOM_Q", "logOM_V1", "logOM_V2",
                 "logOM_MTT", "logOM_CIRC0", "logOM_ALPHA", "logSIGMA_ADD_PK",
                 "logSIGMA_PROP_PK", "logSIGMA_ADD_ANC", "logSIGMA_PROP_ANC"]
        println(io, join(vcat(["method", "start_id", "outer_iterations", "optimizer_converged",
                                "method_value", "canonical_focei_value", "recomputed_method_value", "wall_sec", "cpu_sec",
                                "evaluations", "max_ebe_score", "n_ebe_converged", "failed"], names), ','))
        for row in rows
            prefix = [row.method, row.start_id, row.iterations, row.optimizer_converged,
                      row.method_value, row.canonical_value, row.recomputed_method_value, row.wall_sec, row.cpu_sec,
                      row.evaluations, row.max_score, row.nconv, row.failed]
            println(io, join(vcat(string.(prefix), string.(row.theta)), ','))
        end
    end
end

function fk_main()
    nsubjects = parse(Int, get(ENV, "FK_N_SUBJECTS", string(FK_N_SUBJECTS_DEFAULT)))
    nstarts = parse(Int, get(ENV, "FK_N_STARTS", "5"))
    start_log_sd = parse(Float64, get(ENV, "FK_START_LOG_SD", "0.30"))
    start_log_sd >= 0.0 || error("FK_START_LOG_SD must be nonnegative")
    selected_start_ids = let raw = strip(get(ENV, "FK_START_IDS", ""))
        isempty(raw) ? collect(0:(nstarts - 1)) : parse.(Int, strip.(split(raw, ',')))
    end
    all(start_id -> 0 <= start_id < nstarts, selected_start_ids) ||
        error("FK_START_IDS must contain zero-based start IDs between 0 and $(nstarts - 1)")
    length(unique(selected_start_ids)) == length(selected_start_ids) ||
        error("FK_START_IDS contains duplicate start IDs")
    maxiter = parse(Int, get(ENV, "FK_MAXITER_OUTER", "35"))
    methods = String.(strip.(split(get(ENV, "FK_METHODS", FK_METHODS_DEFAULT), ',')))
    outdir = get(ENV, "FK_OUTDIR", joinpath(@__DIR__, "..", "outputs", "FribergKarlssonFOCEI"))
    mkpath(outdir)

    theta_true = true_theta()
    rng = MersenneTwister(20260822)
    subjects = simulate_population(rng, theta_true; nsubjects=nsubjects)
    lo, hi = fk_bounds()
    starts = fk_sample_starts(theta_true, lo, hi, nstarts; log_sd=start_log_sd)
    writedlm(joinpath(outdir, "friberg_karlsson_start_bank.csv"),
             vcat(reshape(["start_id"; ["theta_$j" for j in 1:P]], 1, :),
                  hcat(collect(0:(nstarts - 1)), starts)), ',')

    println("Friberg--Karlsson matched FOCEI run")
    println("threads=$(nthreads()) design=$(FK_DESIGN) subjects=$nsubjects starts=$nstarts outer_iterations=$maxiter ebe_max_iterations=$(get(ENV, "FK_MAXITER_ETA", "50")) start_log_sd=$start_log_sd")
    println("states=8 structural=9 iiv=$(Q) residual=4 population=$(P) doses_mg=$(join(FK_DOSE_LEVELS_MG, ';')) PK_observations=$(length(PK_TIMES)) ANC_observations=$(length(ANC_TIMES))")
    println("methods=$(join(methods, ',')) output=$outdir")

    # Warm each graph with a two-subject value/gradient calculation before wall
    # time is recorded, avoiding compilation in the production rows.
    for method in methods
        println("warming $method")
        fk_value_grad(method, subjects[1:min(2, end)], starts[1, :])
    end

    rows = NamedTuple[]
    for method in methods, start_id in selected_start_ids
        s = start_id + 1
        println("[$method] start $s/$nstarts")
        row = fk_optimize_one(method, subjects, Vector{Float64}(starts[s, :]), lo, hi; maxiter=maxiter)
        push!(rows, merge(row, (start_id=start_id,)))
        fk_write_rows(joinpath(outdir, "friberg_karlsson_multistart.csv"), rows)
        println("  canonical=$(row.canonical_value) wall=$(row.wall_sec) max_score=$(row.max_score) nconv=$(row.nconv)/$nsubjects")
    end
end

fk_main()
