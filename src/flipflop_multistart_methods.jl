using LinearAlgebra
using Printf
using Random
using Statistics
using ForwardDiff
using ReverseDiff
using Optim
using Base.Threads

const DOSE = 100.0
const TIMES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 24.0, 48.0]
const ETA_DIM = 2
const BIG = 1.0e12
const MIN_POS = 1.0e-12
const RESIDUAL_MODEL = let value = Symbol(lowercase(get(ENV, "FLIPFLOP_JULIA_RESIDUAL_MODEL", "combined")))
    value in (:additive, :combined) ||
        error("FLIPFLOP_JULIA_RESIDUAL_MODEL must be additive or combined; got $value")
    value
end
const PARAM_NAMES = RESIDUAL_MODEL == :combined ?
    ["logka", "logcl", "logv", "logomega_ka", "logomega_v", "logsigma_add", "logsigma_prop"] :
    ["logka", "logcl", "logv", "logomega_ka", "logomega_v", "logsigma"]

struct SubjectData
    y::Vector{Float64}
end

struct FileTime
    low::UInt32
    high::UInt32
end

function process_cpu_seconds()
    if Sys.iswindows()
        handle = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        creation = Ref(FileTime(0, 0))
        exit = Ref(FileTime(0, 0))
        kernel = Ref(FileTime(0, 0))
        user = Ref(FileTime(0, 0))
        ok = ccall((:GetProcessTimes, "kernel32"), Int32,
                   (Ptr{Cvoid}, Ref{FileTime}, Ref{FileTime}, Ref{FileTime}, Ref{FileTime}),
                   handle, creation, exit, kernel, user)
        if ok != 0
            k = (UInt64(kernel[].high) << 32) | UInt64(kernel[].low)
            u = (UInt64(user[].high) << 32) | UInt64(user[].low)
            return Float64(k + u) / 1.0e7
        end
    end
    return time()
end

function base_float(x)
    y = x
    while y isa ForwardDiff.Dual || y isa ReverseDiff.TrackedReal
        if y isa ForwardDiff.Dual
            y = ForwardDiff.value(y)
        else
            y = ReverseDiff.value(y)
        end
    end
    return Float64(y)
end

function pk_conc(t, ka, cl, v)
    ka = max(ka, MIN_POS)
    cl = max(cl, MIN_POS)
    v = max(v, MIN_POS)
    ke = cl / v
    absd = abs(ka - ke)
    base_rate = min(ka, ke)
    base = exp(-base_rate * t)
    x = absd * t
    ratio = if base_float(x) < 1.0e-6
        t - 0.5 * absd * t^2 + (absd^2) * t^3 / 6.0
    else
        -expm1(-x) / max(absd, MIN_POS)
    end
    return (DOSE / v) * ka * base * ratio
end

function unpack_theta(theta)
    if RESIDUAL_MODEL == :combined
        length(theta) == 7 || error("combined residual model requires 7 parameters")
        logka, logcl, logv, logomega_ka, logomega_v, logsigma_add, logsigma_prop = theta
        return (
            logka=logka,
            logcl=logcl,
            logv=logv,
            omega_ka=exp(logomega_ka),
            omega_v=exp(logomega_v),
            sigma_add=exp(logsigma_add),
            sigma_prop=exp(logsigma_prop),
        )
    end

    length(theta) == 6 || error("additive residual model requires 6 parameters")
    logka, logcl, logv, logomega_ka, logomega_v, logsigma = theta
    sigma = exp(logsigma)
    return (
        logka=logka,
        logcl=logcl,
        logv=logv,
        omega_ka=exp(logomega_ka),
        omega_v=exp(logomega_v),
        sigma_add=sigma,
        sigma_prop=zero(sigma),
    )
end

residual_variance(mu, sigma_add, sigma_prop) =
    sigma_add * sigma_add + (sigma_prop * mu) * (sigma_prop * mu)

function residual_variances(predictions, theta)
    pars = unpack_theta(theta)
    return [
        residual_variance(mu, pars.sigma_add, pars.sigma_prop)
        for mu in predictions
    ]
end

function simulate_subjects(theta::Vector{Float64}; n_subj::Int=50, seed::Int=123)
    rng = MersenneTwister(seed)
    pars = unpack_theta(theta)
    subjects = SubjectData[]
    for _ in 1:n_subj
        eta_ka = randn(rng) * pars.omega_ka
        eta_v = randn(rng) * pars.omega_v
        ka = exp(pars.logka + eta_ka)
        cl = exp(pars.logcl)
        v = exp(pars.logv + eta_v)
        y = Float64[]
        for t in TIMES
            mu = pk_conc(t, ka, cl, v)
            sd = sqrt(residual_variance(mu, pars.sigma_add, pars.sigma_prop))
            observation = mu + randn(rng) * sd
            # combined-error experiment uses its stated Gaussian likelihood
            # without censoring or truncating negative simulated observations.
            push!(y, RESIDUAL_MODEL == :additive ? max(observation, 1.0e-8) : observation)
        end
        push!(subjects, SubjectData(y))
    end
    return subjects
end

function read_subjects_long_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty subject data file: $path")
    header = lowercase.(strip.(split(lines[1], ",")))
    id_idx = findfirst(==("id"), header)
    time_idx = findfirst(==("time"), header)
    dv_idx = findfirst(x -> x in ("dv", "y"), header)
    (id_idx === nothing || time_idx === nothing || dv_idx === nothing) &&
        error("subject CSV must contain id,time,dv columns: $path")

    grouped = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = strip.(split(line, ","))
        sid = parse(Int, parts[id_idx])
        tm = parse(Float64, parts[time_idx])
        dv = parse(Float64, parts[dv_idx])
        push!(get!(grouped, sid, Tuple{Float64, Float64}[]), (tm, dv))
    end

    subjects = SubjectData[]
    for sid in sort(collect(keys(grouped)))
        rows = sort(grouped[sid]; by=x -> x[1])
        if length(rows) != length(TIMES)
            error("subject $sid has $(length(rows)) observations; expected $(length(TIMES))")
        end
        push!(subjects, SubjectData([dv for (_, dv) in rows]))
    end
    return subjects
end

function subject_nll(subj::SubjectData, theta, eta)
    pars = unpack_theta(theta)
    ka = exp(pars.logka + eta[1])
    cl = exp(pars.logcl)
    v = exp(pars.logv + eta[2])

    total = zero(ka + cl + v + pars.omega_ka + pars.omega_v +
                 pars.sigma_add + pars.sigma_prop + sum(eta))
    for (j, t) in enumerate(TIMES)
        mu = pk_conc(t, ka, cl, v)
        variance = residual_variance(mu, pars.sigma_add, pars.sigma_prop)
        residual = subj.y[j] - mu
        total += 0.5 * log(variance) + 0.5 * residual * residual / variance
    end
    total += log(pars.omega_ka) + log(pars.omega_v)
    total += 0.5 * (eta[1] / pars.omega_ka)^2
    total += 0.5 * (eta[2] / pars.omega_v)^2
    return total
end

# FOCEI curvature for the conditionally linearized observation model.
# The EBE still solves grad_eta(subject_nll)=0; only the Newton curvature
# and log-determinant correction use this Gauss--Newton matrix.
# For combined error, the covariance-sensitivity term is the positive-definite
# expected-Hessian/NONMEM variant in Almquist et al. (2015), Appendix 2, Eq. 70.
function focei_predictions(subj::SubjectData, theta, eta)
    pars = unpack_theta(theta)
    ka = exp(pars.logka + eta[1])
    cl = exp(pars.logcl)
    v = exp(pars.logv + eta[2])
    return [pk_conc(t, ka, cl, v) for t in TIMES]
end

function add_iiv_precision!(G, theta)
    pars = unpack_theta(theta)
    T = eltype(G)
    G[1, 1] += one(T) / (pars.omega_ka * pars.omega_ka)
    G[2, 2] += one(T) / (pars.omega_v * pars.omega_v)
    return G
end

function foce_curvature_from_components(theta, predictions, J)
    variances = residual_variances(predictions, theta)
    weighted_J = J ./ reshape(variances, :, 1)
    return add_iiv_precision!(transpose(J) * weighted_J, theta)
end

function focei_curvature_from_components(theta, predictions, J, dvariance_deta)
    variances = residual_variances(predictions, theta)
    weighted_J = J ./ reshape(variances, :, 1)
    G = transpose(J) * weighted_J
    weighted_dvariance = dvariance_deta ./ reshape(variances .* variances, :, 1)
    G .+= 0.5 .* (transpose(dvariance_deta) * weighted_dvariance)
    return add_iiv_precision!(G, theta)
end

function foce_curvature(subj::SubjectData, theta, eta)
    J = ForwardDiff.jacobian(e -> focei_predictions(subj, theta, e), eta)
    # Standard noninteraction FOCE freezes the residual covariance at eta=0,
    # while retaining the conditional prediction Jacobian at the current EBE.
    eta_reference = fill(zero(eltype(eta)), length(eta))
    reference_predictions = focei_predictions(subj, theta, eta_reference)
    return foce_curvature_from_components(theta, reference_predictions, J)
end

function focei_curvature(subj::SubjectData, theta, eta)
    predictions = focei_predictions(subj, theta, eta)
    J = ForwardDiff.jacobian(e -> focei_predictions(subj, theta, e), eta)
    pars = unpack_theta(theta)
    row_scale = 2.0 .* (pars.sigma_prop * pars.sigma_prop) .* predictions
    dvariance_deta = J .* reshape(row_scale, :, 1)
    return focei_curvature_from_components(
        theta, predictions, J, dvariance_deta)
end

function primal_float(x)
    y = x
    while y isa ForwardDiff.Dual || y isa ReverseDiff.TrackedReal
        if y isa ForwardDiff.Dual
            y = ForwardDiff.value(y)
        else
            y = ReverseDiff.value(y)
        end
    end
    return Float64(y)
end

is_ad_active(v) = any(x -> x isa ForwardDiff.Dual || x isa ReverseDiff.TrackedReal, v)

function shifted_matrix(H; jitter::Float64=1.0e-6)
    q = size(H, 1)
    T = eltype(H)
    H_float = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        H_float[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, H_float)
        error("non-finite Hessian in shifted_matrix")
    end
    min_eig = minimum(eigvals(Symmetric(H_float)))
    shift = max(jitter, -min_eig + jitter)
    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = (H[i, j] + H[j, i]) / 2
    end
    for i in 1:q
        A[i, i] += T(shift)
    end
    return A
end

function solve_mode_hessian(H, rhs)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    any(!isfinite, Hs) && error("non-finite exact mode Hessian in implicit solve")
    any(!isfinite, rhs) && error("non-finite right-hand side in implicit solve")
    F = cholesky(Symmetric(Hs); check=true)
    return F \ Vector{Float64}(rhs)
end

mutable struct EtaWarmCache
    etas::Dict{Int, Vector{Float64}}
    best_f::Float64
end

EtaWarmCache() = EtaWarmCache(Dict{Int, Vector{Float64}}(), Inf)

function copy_eta_cache(cache::Union{Nothing,EtaWarmCache})
    cache === nothing && return nothing
    return EtaWarmCache(Dict(k => copy(v) for (k, v) in cache.etas), cache.best_f)
end

function eta_start(cache::Union{Nothing,EtaWarmCache}, idx::Int)
    if cache !== nothing && haskey(cache.etas, idx)
        return copy(cache.etas[idx])
    end
    return zeros(ETA_DIM)
end

function maybe_update_eta_cache!(cache::Union{Nothing,EtaWarmCache},
                                 etas, f::Float64, g)
    cache === nothing && return
    isfinite(f) || return
    any(!isfinite, g) && return
    f <= cache.best_f || return
    for i in eachindex(etas)
        eta = Vector{Float64}(etas[i])
        all(isfinite, eta) || return
    end
    for i in eachindex(etas)
        cache.etas[Int(i)] = copy(Vector{Float64}(etas[i]))
    end
    cache.best_f = f
end

function logdet_cholesky_once(H; jitter::Float64)
    T = eltype(H)
    q = size(H, 1)
    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = (H[i, j] + H[j, i]) / 2
    end
    for i in 1:q
        A[i, i] += T(jitter)
    end
    L = zeros(T, q, q)
    for i in 1:q
        for j in 1:i
            s = A[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                if base_float(s) <= 0.0 || !isfinite(base_float(s))
                    error("non-positive Cholesky pivot")
                end
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    return 2 * sum(log(L[i, i]) for i in 1:q)
end

function logdet_cholesky_python_floor(H; jitter::Float64=1.0e-8, max_tries::Int=20,
                                      floor_rel::Float64=1.0e-4, floor_abs::Float64=1.0e-6)
    q = size(H, 1)
    T = eltype(H)
    Hs = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        Hs[i, j] = (H[i, j] + H[j, i]) / 2
    end
    diag_scale = mean(abs(primal_float(Hs[i, i])) for i in 1:q)
    diag_scale = max(diag_scale, 1.0)
    shift = floor_abs + floor_rel * diag_scale + jitter
    for _ in 1:max_tries
        try
            return logdet_cholesky_once(Hs; jitter=shift)
        catch
            shift *= 10.0
        end
    end

    max_need = 0.0
    for i in 1:q
        off = sum(abs(primal_float(Hs[i, j])) for j in 1:q if j != i)
        need = off - primal_float(Hs[i, i]) + floor_abs + floor_rel * diag_scale
        max_need = max(max_need, need)
    end
    return logdet_cholesky_once(Hs; jitter=max(max_need, 0.0) + jitter)
end

function logdet_cholesky(H; jitter::Float64=1.0e-8, max_tries::Int=20)
    logdet_mode = lowercase(get(ENV, "FLIPFLOP_JULIA_LOGDET_MODE", "raw"))
    if logdet_mode in ("python", "python_floor", "python-floor")
        return logdet_cholesky_python_floor(H; jitter=jitter, max_tries=max_tries)
    elseif logdet_mode == "raw"
        # G = Omega^{-1} + J'R^{-1}J is positive definite by construction.
        # Do not perturb the scientific objective when the factorization works.
        return logdet_cholesky_once(H; jitter=0.0)
    end
    error("Unknown FLIPFLOP_JULIA_LOGDET_MODE=$(logdet_mode); use raw or python_floor")
end

function grad_hess_forward(subj::SubjectData, theta, eta)
    f = e -> subject_nll(subj, theta, e)
    return ForwardDiff.gradient(f, eta), focei_curvature(subj, theta, eta), f(eta)
end

fd_rel_step(base::Float64, z::Float64) = base * max(1.0, abs(z))

function focei_curvature_fd(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                            h::Float64=1.0e-3)
    q = length(eta)
    predictions = focei_predictions(subj, theta, eta)
    J = zeros(length(subj.y), q)
    dvariance_deta = zeros(length(subj.y), q)
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        predictions_p = focei_predictions(subj, theta, ep)
        predictions_m = focei_predictions(subj, theta, em)
        J[:, j] .= (predictions_p .- predictions_m) ./ (2.0 * hj)
        dvariance_deta[:, j] .=
            (residual_variances(predictions_p, theta) .-
             residual_variances(predictions_m, theta)) ./ (2.0 * hj)
    end

    return focei_curvature_from_components(
        theta, predictions, J, dvariance_deta)
end

function fd_grad_curvature_eta(subj::SubjectData, theta::Vector{Float64},
                               eta::Vector{Float64}; h::Float64=1.0e-3)
    q = length(eta)
    f0 = subject_nll(subj, theta, eta)
    g = zeros(q)
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        g[j] = (subject_nll(subj, theta, ep) -
                subject_nll(subj, theta, em)) / (2.0 * hj)
    end
    return g, focei_curvature_fd(subj, theta, eta; h=h), f0
end

function fd_newton_step(H, g; jitter::Float64=1.0e-8,
                        floor_rel::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_FLOOR_REL", "1e-6")),
                        floor_abs::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_FLOOR_ABS", "1e-8")),
                        max_step_norm::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_ETA_MAX_STEP", "10.0")))
    q = length(g)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    if any(!isfinite, Hs) || any(!isfinite, g)
        return zeros(q)
    end
    eigs = eigvals(Symmetric(Hs))
    scale = max(1.0, maximum(abs.(eigs)))
    target = max(floor_abs, floor_rel * scale)
    shift = max(0.0, target - minimum(eigs)) + jitter
    Iq = Matrix{Float64}(I, q, q)
    step = nothing
    for k in 0:7
        A = Hs .+ (shift * 10.0^k) .* Iq
        try
            F = cholesky(Symmetric(A); check=false)
            if issuccess(F)
                candidate = F \ Vector{Float64}(g)
                if all(isfinite, candidate)
                    step = candidate
                    break
                end
            end
        catch
        end
    end
    step === nothing && (step = Vector{Float64}(g))
    nrm = norm(step)
    if !isfinite(nrm)
        return zeros(q)
    end
    if nrm > max_step_norm
        step .*= max_step_norm / (nrm + 1.0e-12)
    end
    return step
end

function safe_newton_step(H, g; max_step_norm::Float64=10.0, jitter::Float64=1.0e-8)
    A = Matrix{Float64}(shifted_matrix(H; jitter=jitter))
    step = A \ Vector{Float64}(g)
    nrm = norm(step)
    if isfinite(nrm) && nrm > max_step_norm
        step .*= max_step_norm / (nrm + 1.0e-12)
    end
    return step
end

function unroll_newton_step(H, g; jitter::Float64=1.0e-6, floor_rel::Float64=1.0e-4,
                            floor_abs::Float64=1.0e-6, max_step_norm::Float64=3.0)
    q = length(g)
    T = promote_type(eltype(H), eltype(g))
    Hs = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        Hs[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, Hs)
        error("non-finite Hessian in unrolled Newton step")
    end

    scale = max(mean(abs.(diag(Hs))), 1.0)
    shift = floor_abs + floor_rel * scale + jitter
    Iq = Matrix{Float64}(I, q, q)
    A_float = Hs + shift .* Iq
    shift_ok = false
    for _ in 1:8
        try
            cholesky(Symmetric(A_float))
            shift_ok = true
            break
        catch
        end
        shift *= 10.0
        A_float = Hs + shift .* Iq
    end

    if !shift_ok
        return g
    end

    A = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        A[i, j] = T((H[i, j] + H[j, i]) / 2)
    end
    for i in 1:q
        A[i, i] += T(shift)
    end
    step = A \ T.(g)

    nrm = sqrt(sum(abs2, [primal_float(s) for s in step]))
    if !isfinite(nrm)
        error("non-finite unrolled Newton step")
    end
    if nrm > max_step_norm
        step = step .* (max_step_norm / (sqrt(sum(abs2, step)) + 1.0e-12))
    end
    return step
end

function eta_mode_newton(subj::SubjectData, theta::Vector{Float64}; maxiter::Int=20,
                         tol::Float64=1.0e-8, backend::Symbol=:forward,
                         eps_eta::Float64=1.0e-3,
                         eta0::Union{Nothing,Vector{Float64}}=nothing)
    eta = eta0 === nothing ? zeros(ETA_DIM) : copy(eta0)
    f_curr = subject_nll(subj, theta, eta)
    grad_norm = Inf
    converged = false
    for _ in 1:maxiter
        g, H, f0 = backend == :fd ? fd_grad_curvature_eta(subj, theta, eta; h=eps_eta) : grad_hess_forward(subj, theta, eta)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        if grad_norm < tol
            f_curr = f0
            converged = true
            break
        end
        step = backend == :fd ? fd_newton_step(H, g) : safe_newton_step(H, g)
        alpha = 1.0
        accepted = false
        for _ls in 1:12
            trial = eta .- alpha .* step
            f_try = subject_nll(subj, theta, trial)
            if isfinite(f_try) && f_try <= f0 + 1.0e-10
                eta = trial
                f_curr = f_try
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    final_g = backend == :fd ? fd_grad_curvature_eta(subj, theta, eta; h=eps_eta)[1] :
                               ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta)
    grad_norm = norm(final_g)
    convergence_tol = backend == :fd ? max(tol, 10.0 * eps_eta^2) : tol
    converged = isfinite(grad_norm) && grad_norm < convergence_tol
    return eta, Float64(subject_nll(subj, theta, eta)), grad_norm, converged
end

function eta_mode_bfgs(subj::SubjectData, theta::Vector{Float64}; maxiter::Int=20,
                       tol::Float64=1.0e-8, backend::Symbol=:forward,
                       eps_eta::Float64=1.0e-4,
                       eta0::Union{Nothing,Vector{Float64}}=nothing)
    cache_eta = zeros(ETA_DIM)
    cache_valid = false
    cache_f = Inf
    cache_g = zeros(ETA_DIM)
    function eval_eta!(eta)
        eta_vec = Vector{Float64}(eta)
        if cache_valid && all(cache_eta .== eta_vec)
            return
        end
        f0 = subject_nll(subj, theta, eta_vec)
        g = backend == :fd ? fd_grad_curvature_eta(subj, theta, eta_vec; h=eps_eta)[1] :
                              ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta_vec)
        cache_eta = eta_vec
        cache_f = Float64(f0)
        cache_g = Vector{Float64}(g)
        cache_valid = true
    end
    f(eta) = begin
        eval_eta!(eta)
        cache_f
    end
    function g!(G, eta)
        eval_eta!(eta)
        G .= cache_g
        return G
    end
    eta_initial = eta0 === nothing ? zeros(ETA_DIM) : copy(eta0)
    result = optimize(f, g!, eta_initial, BFGS(),
                      Optim.Options(iterations=maxiter, g_tol=tol, f_reltol=1.0e-10, show_trace=false))
    eta_hat = Vector{Float64}(Optim.minimizer(result))
    final_g = backend == :fd ? fd_grad_curvature_eta(subj, theta, eta_hat; h=eps_eta)[1] :
                               ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta_hat)
    grad_norm = norm(final_g)
    return eta_hat, Float64(subject_nll(subj, theta, eta_hat)), grad_norm, grad_norm < max(tol, 1.0e-6)
end

function focei_subject_fixed_eta(subj::SubjectData, theta, eta)
    f = e -> subject_nll(subj, theta, e)
    H = focei_curvature(subj, theta, eta)
    hi = f(eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky(H))
end

function focei_subject_fixed_eta_fd(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                     eps_eta::Float64=1.0e-3)
    hi = subject_nll(subj, theta, eta)
    ld = strict_fd_logdet(focei_curvature_fd(subj, theta, eta; h=eps_eta))
    return isfinite(hi) && isfinite(ld) ? 2.0 * (hi + 0.5 * ld) : BIG
end
function strict_fd_logdet(H; jitter::Float64=0.0)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    any(!isfinite, Hs) && return Inf
    q = size(Hs, 1)
    F = cholesky(Symmetric(Hs .+ jitter .* Matrix{Float64}(I, q, q)); check=false)
    issuccess(F) || return Inf
    return 2.0 * sum(log, diag(F.L))
end

function solve_all_etas(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter::Int=20,
                        backend::Symbol=:forward, eps_eta::Float64=1.0e-3,
                        solver::Symbol=:newton,
                        eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    # BitVector writes are not independent across threads because flags share
    # packed machine words.
    converged = Vector{Bool}(undef, length(subjects))
    @threads for i in eachindex(subjects)
        eta0 = eta_start(eta_cache, Int(i))
        eta, _, gn, cvg = solver == :bfgs ?
            eta_mode_bfgs(subjects[i], theta; maxiter=maxiter, backend=backend, eps_eta=eps_eta, eta0=eta0) :
            eta_mode_newton(subjects[i], theta; maxiter=maxiter, backend=backend, eps_eta=eps_eta, eta0=eta0)
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function outer_gradient(f, theta::Vector{Float64})
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_OUTER_AD_MODE", "forward"))
    if mode in ("reverse", "backward")
        return ReverseDiff.gradient(f, theta)
    elseif mode == "forward"
        return ForwardDiff.gradient(f, theta)
    end
    error("Unknown FLIPFLOP_JULIA_OUTER_AD_MODE=$(mode); use reverse or forward")
end

function parse_eta_solver()
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_ETA_SOLVER", "newton"))
    mode in ("newton", "hessian", "second_order", "second-order") && return :newton
    mode in ("bfgs", "gradient", "gradient_only", "gradient-only") && return :bfgs
    error("Unknown FLIPFLOP_JULIA_ETA_SOLVER=$(mode); use newton or bfgs")
end

function use_eta_cache()
    mode = lowercase(get(ENV, "FLIPFLOP_JULIA_USE_ETA_CACHE", "true"))
    return !(mode in ("0", "false", "no", "off"))
end

function stop_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                         eta_solver::Symbol=:newton,
                         eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta = etas[i]
        ftheta = x -> focei_subject_fixed_eta(subj, x, eta)
        vals[i] = Float64(ftheta(theta))
        grads[i] = outer_gradient(ftheta, theta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function ad_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                             eta_solver::Symbol=:newton,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = Float64(focei_subject_fixed_eta(subjects[i], theta, etas[i]))
    end
    return sum(vals), max_eta_grad, n_conv
end

function safe_ad_population_evaluation(subjects::Vector{SubjectData}, theta::Vector{Float64};
                                       maxiter_eta::Int=20, eta_solver::Symbol=:newton)
    try
        return ad_population_value(subjects, theta; maxiter_eta=maxiter_eta, eta_solver=eta_solver)
    catch err
        @warn "AD endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN, NaN, 0
    end
end

safe_ad_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64};
                         maxiter_eta::Int=20, eta_solver::Symbol=:newton) =
    safe_ad_population_evaluation(subjects, theta; maxiter_eta=maxiter_eta,
                                  eta_solver=eta_solver)[1]

function full_implicit_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    ofv = focei_subject_fixed_eta(subj, theta, eta)
    dh_dx = ForwardDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dx = ForwardDiff.gradient(x -> logdet_cholesky(focei_curvature(subj, x, eta)), theta)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky(focei_curvature(subj, theta, e)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(
        x -> dot(ForwardDiff.gradient(e -> subject_nll(subj, x, e), eta), lambda),
        theta,
    )
    grad = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- term_c)
    return Float64(ofv), Vector{Float64}(grad)
end

function full_implicit_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                                  eta_solver::Symbol=:newton,
                                  eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = full_implicit_subject_value_grad(subjects[i], theta, etas[i])
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

# Solve the primal EBE to convergence, detach it, and differentiate one exact
# Newton update. At an exact root this has the same local mode sensitivity as
# implicit differentiation, while retaining a distinct one-step AD graph.
function one_step_newton_subject_value(subj::SubjectData, theta, eta_star::Vector{Float64})
    eta = eltype(theta).(eta_star)
    g = ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta)
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    step = H \ g
    return focei_subject_fixed_eta(subj, theta, eta .- step)
end

function one_step_newton_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64};
                                    maxiter_eta::Int=20,
                                    eta_solver::Symbol=:newton,
                                    eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    eta_solver == :newton || error("one-step Newton requires the exact-Hessian Newton EBE solver")
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    if n_conv != length(subjects)
        grad_norms = zeros(length(subjects))
        converged = Vector{Bool}(undef, length(subjects))
        @threads for i in eachindex(subjects)
            score = ForwardDiff.gradient(e -> subject_nll(subjects[i], theta, e), etas[i])
            if norm(score) < 1.0e-8
                grad_norms[i] = norm(score)
                converged[i] = true
            else
                eta, _, gn, cvg = eta_mode_newton(subjects[i], theta; maxiter=max(300, maxiter_eta),
                                                   tol=1.0e-10, backend=:forward,
                                                   eta0=etas[i])
                etas[i] = eta
                grad_norms[i] = gn
                converged[i] = cvg || gn < 1.0e-8
            end
        end
        max_eta_grad = maximum(grad_norms)
        n_conv = count(converged)
    end
    n_conv == length(subjects) || error("one-step Newton requires converged primal EBEs")
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta_star = etas[i]
        ftheta = x -> one_step_newton_subject_value(subj, x, eta_star)
        vals[i] = Float64(ftheta(theta))
        grads[i] = outer_gradient(ftheta, theta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end
function eta_newton_unroll(subj::SubjectData, theta, eta0::Vector{Float64}; steps::Int=20)
    T = eltype(theta)
    eta = T.(eta0)
    for _ in 1:steps
        g, H, hi = grad_hess_forward(subj, theta, eta)
        if !isfinite(primal_float(hi)) || any(!isfinite, [primal_float(s) for s in g])
            error("non-finite unrolled eta gradient")
        end
        step = unroll_newton_step(H, g)
        step_norm = sqrt(sum(abs2, [primal_float(s) for s in step]))

        theta_pr = [primal_float(s) for s in theta]
        eta_pr = [primal_float(s) for s in eta]
        step_pr = [primal_float(s) for s in step]
        f0 = subject_nll(subj, theta_pr, eta_pr)
        if !isfinite(Float64(f0))
            error("non-finite unrolled eta objective")
        end
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial_pr = eta_pr .- alpha .* step_pr
            f_try = subject_nll(subj, theta_pr, trial_pr)
            if isfinite(Float64(f_try)) && Float64(f_try) <= Float64(f0) + 1.0e-10
                eta = eta .- alpha .* step
                accepted = true
                break
            end
            alpha *= 0.5
        end
        if !accepted
            break
        end
    end
    return eta
end

function full_unroll_subject_value(subj::SubjectData, theta, eta0::Vector{Float64}; steps::Int=20)
    eta = eta_newton_unroll(subj, theta, eta0; steps=steps)
    return focei_subject_fixed_eta(subj, theta, eta)
end

function full_unroll_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; steps::Int=20,
                                eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    vals = zeros(length(subjects))
    grads = [zeros(length(theta)) for _ in subjects]
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    diagnostic_tol = parse(Float64, get(ENV, "FLIPFLOP_JULIA_FULL_UNROLL_ETA_TOL", "1e-6"))
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta0 = eta_start(eta_cache, Int(i))
        eta = eta_newton_unroll(subj, theta, eta0; steps=steps)
        etas[i] = [primal_float(s) for s in eta]
        final_score = ForwardDiff.gradient(e -> subject_nll(subj, theta, e), etas[i])
        grad_norms[i] = norm(final_score)
        ftheta = x -> full_unroll_subject_value(subj, x, eta0; steps=steps)
        vals[i] = Float64(ftheta(theta))
        grads[i] = outer_gradient(ftheta, theta)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_grad)
    return total, total_grad, maximum(grad_norms), count(<(diagnostic_tol), grad_norms)
end

function fd_population_value(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                             eps_eta::Float64=1.0e-3, eta_solver::Symbol=:newton,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta, backend=:fd,
                                                eps_eta=eps_eta, solver=eta_solver,
                                                eta_cache=eta_cache)
    require_convergence = lowercase(get(ENV, "FLIPFLOP_JULIA_FD_REQUIRE_PERTURBED_ETA_CONVERGENCE", "true")) in
                          ("1", "true", "yes", "on")
    if require_convergence && n_conv != length(subjects)
        error("FD perturbed EBE solve converged for $n_conv/$(length(subjects)); max score norm=$max_eta_grad")
    end
    return fd_population_value_from_etas(subjects, theta, etas; eps_eta=eps_eta)
end

function fd_population_value_from_etas(subjects::Vector{SubjectData}, theta::Vector{Float64}, etas;
                                       eps_eta::Float64=1.0e-3)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = focei_subject_fixed_eta_fd(subjects[i], theta, etas[i]; eps_eta=eps_eta)
    end
    return sum(vals)
end

function fd_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                       eps::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_EPS_THETA", "1e-3")),
                       eps_eta::Float64=parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_EPS_ETA", "1e-3")),
                       eta_solver::Symbol=:newton,
                       eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:fd, eps_eta=eps_eta,
                                                solver=eta_solver, eta_cache=eta_cache)
    f0 = fd_population_value_from_etas(subjects, theta, etas; eps_eta=eps_eta)
    base_cache = EtaWarmCache()
    for i in eachindex(subjects)
        base_cache.etas[Int(i)] = copy(etas[i])
    end
    base_cache.best_f = isfinite(f0) ? f0 : Inf
    g = zeros(length(theta))
    lo, hi = theta_bounds()
    @threads for j in eachindex(theta)
        xp = copy(theta)
        xm = copy(theta)
        h = eps * max(1.0, abs(theta[j]))
        xp[j] = min(max(theta[j] + h, lo[j]), hi[j])
        xm[j] = min(max(theta[j] - h, lo[j]), hi[j])
        step_p = xp[j] - theta[j]
        step_m = theta[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            g[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            f1 = fd_population_value(subjects, xone; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            isfinite(f1) || error("non-finite one-sided FD population objective for parameter $j")
            g[j] = (f1 - f0) / step
        else
            fp = fd_population_value(subjects, xp; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            fm = fd_population_value(subjects, xm; maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_solver=eta_solver, eta_cache=copy_eta_cache(base_cache))
            if isfinite(fp) && isfinite(fm)
                g[j] = (fp - fm) / (step_p + step_m)
            elseif isfinite(fp)
                g[j] = (fp - f0) / step_p
            elseif isfinite(fm)
                g[j] = (f0 - fm) / step_m
            else
                error("non-finite two-sided FD population objectives for parameter $j")
            end
        end
    end
    fd_cache_tol = parse(Float64, get(ENV, "FLIPFLOP_JULIA_FD_CACHE_MAX_ETA_GRAD", "1e-3"))
    if n_conv == length(subjects) || (isfinite(max_eta_grad) && max_eta_grad <= fd_cache_tol)
        maybe_update_eta_cache!(eta_cache, etas, Float64(f0), g)
    end
    return f0, g, max_eta_grad, n_conv
end

function theta_bounds()
    if RESIDUAL_MODEL == :combined
        lo = log.([1.0e-3, 0.05, 1.0, 0.01, 0.01, 0.003, 0.01])
        hi = log.([2.0, 30.0, 300.0, 1.0, 1.0, 1.0, 1.0])
        return lo, hi
    end
    lo = log.([1.0e-3, 0.05, 1.0, 0.01, 0.01, 0.01])
    hi = log.([2.0, 30.0, 300.0, 1.0, 1.0, 5.0])
    return lo, hi
end

function clip_to_bounds(theta::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    return min.(max.(theta, lo), hi)
end

function sample_starts(n::Int, theta_true1::Vector{Float64}, theta_true2::Vector{Float64},
                       lo::Vector{Float64}, hi::Vector{Float64}; seed::Int=20260322)
    rng = MersenneTwister(seed)
    starts = Matrix{Float64}(undef, n, length(theta_true1))
    scale = RESIDUAL_MODEL == :combined ?
        [0.5, 0.15, 0.5, 0.15, 0.15, 0.20, 0.20] : [0.5, 0.15, 0.5, 0.15, 0.15, 0.15]
    for s in 1:n
        center = rand(rng) < 0.5 ? theta_true1 : theta_true2
        starts[s, :] = clip_to_bounds(center .+ randn(rng, length(center)) .* scale, lo, hi)
    end
    return starts
end

function read_start_bank_csv(path::AbstractString, lo::Vector{Float64}, hi::Vector{Float64}; n_starts::Int)
    lines = readlines(path)
    isempty(lines) && error("empty start-bank file: $path")
    header = lowercase.(strip.(split(lines[1], ",")))
    idxs = [findfirst(==(name), header) for name in PARAM_NAMES]
    any(isnothing, idxs) && error("start bank must contain $(join(PARAM_NAMES, ", ")): $path")

    rows = Vector{Vector{Float64}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = strip.(split(line, ","))
        theta = [parse(Float64, parts[Int(idx)]) for idx in idxs]
        push!(rows, clip_to_bounds(theta, lo, hi))
        length(rows) >= n_starts && break
    end
    length(rows) < n_starts && error("start bank $path has $(length(rows)) rows; expected at least $n_starts")
    starts = Matrix{Float64}(undef, n_starts, length(PARAM_NAMES))
    for i in 1:n_starts
        starts[i, :] .= rows[i]
    end
    return starts
end

function method_evaluator(method::String, subjects::Vector{SubjectData}; maxiter_eta::Int=20,
                          full_unroll_steps::Int=20, eta_solver::Symbol=:newton)
    eta_cache = use_eta_cache() ? EtaWarmCache() : nothing
    if method == "STOP"
        return theta -> stop_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                        eta_solver=eta_solver, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT"
        return theta -> full_implicit_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                                 eta_solver=eta_solver, eta_cache=eta_cache)
    elseif method == "FULL_UNROLL_1NEWTON"
        return theta -> one_step_newton_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                                   eta_solver=eta_solver, eta_cache=eta_cache)
    elseif method == "FULL_UNROLL"
        eta_solver == :newton || error("FULL_UNROLL currently requires the differentiable Newton unroll eta solver")
        use_unroll_cache = lowercase(get(ENV, "FLIPFLOP_JULIA_FULL_UNROLL_USE_ETA_CACHE", "false")) in
                           ("1", "true", "yes", "on")
        unroll_cache = use_unroll_cache ? eta_cache : nothing
        return theta -> full_unroll_value_grad(subjects, theta; steps=full_unroll_steps,
                                               eta_cache=unroll_cache)
    elseif method == "FD"
        return theta -> fd_value_grad(subjects, theta; maxiter_eta=maxiter_eta,
                                      eta_solver=eta_solver, eta_cache=eta_cache)
    end
    error("unknown method: $method")
end

mutable struct EvalCache
    valid::Bool
    x::Vector{Float64}
    f::Float64
    g::Vector{Float64}
    max_eta_grad::Float64
    n_converged::Int
    evals::Int
    failed::Bool
    has_finite::Bool
    last_finite_x::Vector{Float64}
    last_finite_f::Float64
end

function EvalCache(p::Int)
    return EvalCache(false, zeros(p), Inf, zeros(p), NaN, 0, 0,
                     false, false, zeros(p), Inf)
end

struct OptimFallbackResult
    x::Vector{Float64}
    iterations::Int
    converged::Bool
end

Optim.minimizer(r::OptimFallbackResult) = r.x
Optim.iterations(r::OptimFallbackResult) = r.iterations
Optim.converged(r::OptimFallbackResult) = r.converged

function ensure_cache!(cache::EvalCache, evaluator, x)
    xx = Vector{Float64}(x)
    if cache.valid && length(cache.x) == length(xx) && all(cache.x .== xx)
        return
    end
    try
        f, g, max_eta_grad, n_conv = evaluator(xx)
        if !isfinite(f) || any(!isfinite, g)
            error("non-finite objective or gradient")
        end
        cache.x = xx
        cache.f = Float64(f)
        cache.g = Vector{Float64}(g)
        cache.max_eta_grad = Float64(max_eta_grad)
        cache.n_converged = Int(n_conv)
        cache.failed = false
        cache.has_finite = true
        cache.last_finite_x = copy(xx)
        cache.last_finite_f = cache.f
    catch err
        cache.x = xx
        if cache.has_finite && length(cache.last_finite_x) == length(xx)
            d = xx .- cache.last_finite_x
            cache.f = BIG + 1.0e6 * sum(abs2, d)
            cache.g = 2.0e6 .* d
        else
            cache.f = BIG + sum(abs2, xx)
            cache.g = 2.0 .* xx
        end
        cache.max_eta_grad = NaN
        cache.n_converged = 0
        cache.failed = true
        @warn "method evaluation failed; returning smooth penalty" exception=typeof(err)
    end
    cache.valid = true
    cache.evals += 1
end

function optimize_one(method::String, subjects::Vector{SubjectData}, theta0::Vector{Float64},
                      lo::Vector{Float64}, hi::Vector{Float64}; maxiter_eta::Int=20,
                      full_unroll_steps::Int=20, maxiter_outer::Int=50,
                      eta_solver::Symbol=:newton)
    evaluator = method_evaluator(method, subjects; maxiter_eta=maxiter_eta,
                                 full_unroll_steps=full_unroll_steps,
                                 eta_solver=eta_solver)
    cache = EvalCache(length(theta0))
    bound_mode = lowercase(get(ENV, "FLIPFLOP_JULIA_BOUND_MODE", "fminbox"))
    function bounds_penalty(x)
        xx = Vector{Float64}(x)
        below = max.(lo .- xx, 0.0)
        above = max.(xx .- hi, 0.0)
        d = above .- below
        if any(!iszero, d)
            return true, BIG + 1.0e6 * sum(abs2, d), 2.0e6 .* d
        end
        return false, 0.0, zeros(length(xx))
    end
    f_box(x) = begin
        ensure_cache!(cache, evaluator, Vector{Float64}(x))
        cache.f
    end
    function g_box!(G, x)
        ensure_cache!(cache, evaluator, Vector{Float64}(x))
        G .= cache.g
        return G
    end
    f_penalty(x) = begin
        outside, fp, _ = bounds_penalty(x)
        outside && return fp
        ensure_cache!(cache, evaluator, x)
        cache.f
    end
    function g_penalty!(G, x)
        outside, _, gp = bounds_penalty(x)
        if outside
            G .= gp
            return G
        end
        ensure_cache!(cache, evaluator, x)
        G .= cache.g
        return G
    end

    GC.gc()
    wall0 = time_ns()
    cpu0 = process_cpu_seconds()
    theta0_box = min.(max.(theta0, lo), hi)
    linesearch = lowercase(get(ENV, "FLIPFLOP_JULIA_LINESEARCH", "hagerzhang"))
    lbfgs = linesearch == "backtracking" ? LBFGS(linesearch=Optim.LineSearches.BackTracking()) : LBFGS()
    result = try
        if bound_mode == "penalty"
            optimize(f_penalty, g_penalty!, theta0_box, lbfgs,
                     Optim.Options(iterations=maxiter_outer, outer_iterations=maxiter_outer, g_tol=1.0e-6,
                                   f_reltol=1.0e-8, show_trace=false))
        elseif bound_mode == "fminbox"
            optimize(f_box, g_box!, lo, hi, theta0_box, Fminbox(lbfgs),
                     Optim.Options(iterations=maxiter_outer, outer_iterations=maxiter_outer, g_tol=1.0e-6,
                                   f_reltol=1.0e-8, show_trace=false))
        else
            error("Unknown FLIPFLOP_JULIA_BOUND_MODE=$(bound_mode); use fminbox or penalty")
        end
    catch err
        @warn "outer optimizer failed; returning last finite point" method=method exception=typeof(err)
        fallback_x = cache.has_finite ? copy(cache.last_finite_x) : copy(theta0_box)
        cache.failed = true
        OptimFallbackResult(fallback_x, 0, false)
    end
    wall = (time_ns() - wall0) / 1.0e9
    cpu = process_cpu_seconds() - cpu0
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result)), lo), hi)
    ensure_cache!(cache, evaluator, theta_hat)
    return result, cache, wall, cpu
end

function optimizer_termination(result, cache::EvalCache)
    cache.failed && return "evaluation_exception_fallback"
    result isa OptimFallbackResult && return "optimizer_exception_fallback"
    flags = String[]
    Optim.x_converged(result) && push!(flags, "x_converged")
    Optim.f_converged(result) && push!(flags, "f_converged")
    Optim.g_converged(result) && push!(flags, "g_converged")
    if Optim.converged(result)
        return isempty(flags) ? "converged" : join(flags, "+")
    end
    if Optim.iteration_limit_reached(result)
        return "iteration_limit"
    end
    return isempty(flags) ? "not_converged" : "not_converged+" * join(flags, "+")
end

model_label() = RESIDUAL_MODEL == :combined ? "flipflop_combined" : "flipflop"
output_prefix() = RESIDUAL_MODEL == :combined ? "flipflop_combined" : "flipflop"

function natural_parameter_names()
    if RESIDUAL_MODEL == :combined
        return ["ka_pop", "cl_pop", "v_pop", "omega_ka", "omega_v", "sigma_add", "sigma_prop"]
    end
    return ["ka_pop", "cl_pop", "v_pop", "omega_ka", "omega_v", "sigma"]
end

function write_results(path::AbstractString, rows)
    theta_headers = ["theta_" * name for name in PARAM_NAMES]
    open(path, "w") do io
        header = vcat(
            [
                "model", "residual_model", "implementation", "method", "start_id",
                "n_subjects", "maxiter_eta", "eta_solver", "full_unroll_steps",
                "outer_iterations", "outer_converged", "termination", "success",
                "objective", "objective_ad_eval",
                "objective_stop_eval", "ad_eval_max_eta_grad_norm",
                "ad_eval_n_eta_converged", "wall_sec", "cpu_sec", "method_eval_count",
                "max_eta_grad_norm", "n_eta_converged",
            ],
            theta_headers,
            natural_parameter_names(),
        )
        println(io, join(header, ","))
        for r in rows
            theta = r.theta
            vals = vcat(
                Any[
                    r.model, r.residual_model, r.implementation, r.method, r.start_id,
                    r.n_subjects, r.maxiter_eta, r.eta_solver, r.full_unroll_steps,
                    r.outer_iterations, r.outer_converged, r.termination, r.success,
                    r.objective, r.objective_ad_eval,
                    r.objective_stop_eval, r.ad_eval_max_eta_grad_norm,
                    r.ad_eval_n_eta_converged, r.wall_sec, r.cpu_sec,
                    r.method_eval_count, r.max_eta_grad_norm, r.n_eta_converged,
                ],
                theta,
                exp.(theta),
            )
            println(io, join(string.(vals), ","))
        end
    end
end

function write_start_bank(path::AbstractString, starts::Matrix{Float64})
    open(path, "w") do io
        header = vcat(
            ["model", "residual_model", "start_id"],
            PARAM_NAMES,
            natural_parameter_names(),
        )
        println(io, join(header, ","))
        for s in 1:size(starts, 1)
            theta = Vector{Float64}(starts[s, :])
            vals = vcat(
                [model_label(), string(RESIDUAL_MODEL), string(s - 1)],
                string.(theta),
                string.(exp.(theta)),
            )
            println(io, join(vals, ","))
        end
    end
end

function write_design_metadata(path::AbstractString, theta_true1::Vector{Float64},
                               theta_true2::Vector{Float64}, n_subj::Int; data_seed::Int=123)
    open(path, "w") do io
        header = vcat(
            ["design", "residual_model", "n_subjects", "data_seed"],
            PARAM_NAMES,
            natural_parameter_names(),
        )
        println(io, join(header, ","))
        for (label, theta) in (("flipflop_basin_1", theta_true1), ("flipflop_basin_2", theta_true2))
            vals = vcat(
                [label, string(RESIDUAL_MODEL), string(n_subj), string(data_seed)],
                string.(theta),
                string.(exp.(theta)),
            )
            println(io, join(vals, ","))
        end
    end
end
function write_nonmem_data(path::AbstractString, subjects::Vector{SubjectData})
    open(path, "w") do io
        println(io, "ID TIME AMT DV EVID MDV CMT")
        for (i, subj) in enumerate(subjects)
            println(io, join((i, 0.0, DOSE, 0.0, 1, 1, 1), " "))
            for (t, y) in zip(TIMES, subj.y)
                println(io, join((i, t, 0.0, y, 0, 0, 2), " "))
            end
        end
    end
end

function split_methods(s::String)
    return [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

function main()
    n_subj = parse(Int, get(ENV, "FLIPFLOP_JULIA_N_SUBJ", "50"))
    n_starts = parse(Int, get(ENV, "FLIPFLOP_JULIA_N_STARTS", "100"))
    maxiter_eta = parse(Int, get(ENV, "FLIPFLOP_JULIA_MAXITER_ETA", "20"))
    maxiter_outer = parse(Int, get(ENV, "FLIPFLOP_JULIA_MAXITER_OUTER", "50"))
    full_unroll_steps = parse(Int, get(ENV, "FLIPFLOP_JULIA_FULL_UNROLL_STEPS", string(maxiter_eta)))
    eta_solver = parse_eta_solver()
    methods = split_methods(get(ENV, "FLIPFLOP_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL_1NEWTON,STOP,FD"))
    outdir = get(ENV, "FLIPFLOP_JULIA_OUTDIR", joinpath(@__DIR__, "tables"))
    mkpath(outdir)
    prefix = output_prefix()
    outpath = joinpath(outdir, "$(prefix)_julia_multistart_methods.csv")
    starts_path = joinpath(outdir, "$(prefix)_julia_start_bank.csv")
    nonmem_data_path = joinpath(outdir, "$(prefix)_nonmem.dat")
    design_path = joinpath(outdir, "$(prefix)_design.csv")

    theta_true1 = RESIDUAL_MODEL == :combined ?
        log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.03, 0.15]) : log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.1])
    theta_true2 = RESIDUAL_MODEL == :combined ?
        log.([3.0 / 20.0, 3.0, 3.0 / 0.05, 0.1, 0.1, 0.03, 0.15]) : log.([3.0 / 20.0, 3.0, 3.0 / 0.05, 0.1, 0.1, 0.1])
    lo, hi = theta_bounds()
    subjects_csv = get(ENV, "FLIPFLOP_JULIA_SUBJECTS_CSV", "")
    starts_csv = get(ENV, "FLIPFLOP_JULIA_STARTS_CSV", "")
    data_seed = parse(Int, get(ENV, "FLIPFLOP_JULIA_DATA_SEED", "123"))
    start_seed = parse(Int, get(ENV, "FLIPFLOP_JULIA_START_SEED", "20260322"))
    subjects = isempty(subjects_csv) ?
        simulate_subjects(theta_true1; n_subj=n_subj, seed=data_seed) : read_subjects_long_csv(subjects_csv)
    if !isempty(subjects_csv)
        n_subj = length(subjects)
    end
    starts = isempty(starts_csv) ?
        sample_starts(n_starts, theta_true1, theta_true2, lo, hi; seed=start_seed) :
        read_start_bank_csv(starts_csv, lo, hi; n_starts=n_starts)
    write_start_bank(starts_path, starts)
    write_nonmem_data(nonmem_data_path, subjects)
    write_design_metadata(design_path, theta_true1, theta_true2, n_subj; data_seed=data_seed)

    println("Flip-flop Julia multistart methods")
    println("threads=", nthreads(), " subjects=", n_subj, " starts=", n_starts)
    println("residual_model=", RESIDUAL_MODEL, " true_parameters=", join(exp.(theta_true1), ","))
    println("methods=", join(methods, ","), " maxiter_eta=", maxiter_eta,
            " full_unroll_steps=", full_unroll_steps, " maxiter_outer=", maxiter_outer)
    println("eta_solver=", eta_solver, " eta_dim=", ETA_DIM)
    println("output=", outpath)
    println("starts=", starts_path)
    println("nonmem_data=", nonmem_data_path)
    println("design=", design_path)
    !isempty(subjects_csv) && println("matched_subjects_csv=", subjects_csv)
    !isempty(starts_csv) && println("matched_starts_csv=", starts_csv)

    # Warm the compiled method paths outside reported timings.
    for method in methods
        println("warming ", method)
        evaluator = method_evaluator(method, subjects; maxiter_eta=min(maxiter_eta, 3),
                                     full_unroll_steps=min(full_unroll_steps, 3),
                                     eta_solver=eta_solver)
        try
            evaluator(Vector{Float64}(starts[1, :]))
        catch err
            @warn "warm-up evaluation failed; continuing to the configured production evaluator" method=method exception=typeof(err)
        end
    end

    rows = NamedTuple[]
    for method in methods
        for s in 1:n_starts
            theta0 = Vector{Float64}(starts[s, :])
            @printf("\n[%s] start %d/%d\n", method, s, n_starts)
            result, cache, wall, cpu = optimize_one(method, subjects, theta0, lo, hi;
                                                    maxiter_eta=maxiter_eta,
                                                    full_unroll_steps=full_unroll_steps,
                                                    maxiter_outer=maxiter_outer,
                                                    eta_solver=eta_solver)
            theta_hat = Vector{Float64}(Optim.minimizer(result))
            ad_eval, ad_eval_max_eta_grad, ad_eval_n_conv =
                safe_ad_population_evaluation(subjects, theta_hat; maxiter_eta=maxiter_eta,
                                              eta_solver=eta_solver)
            stop_eval = ad_eval
            row = (
                model=model_label(),
                residual_model=string(RESIDUAL_MODEL),
                implementation=method == "FD" ? "julia_finite_difference" : "julia_forward_eta",
                method=method,
                start_id=s - 1,
                n_subjects=n_subj,
                maxiter_eta=maxiter_eta,
                eta_solver=string(eta_solver),
                full_unroll_steps=full_unroll_steps,
                outer_iterations=Optim.iterations(result),
                outer_converged=Optim.converged(result),
                termination=optimizer_termination(result, cache),
                success=Optim.converged(result) && !cache.failed && isfinite(ad_eval) &&
                        isfinite(cache.max_eta_grad) && cache.n_converged == n_subj &&
                        isfinite(ad_eval_max_eta_grad) && ad_eval_n_conv == n_subj,
                objective=cache.f,
                objective_ad_eval=ad_eval,
                objective_stop_eval=stop_eval,
                ad_eval_max_eta_grad_norm=ad_eval_max_eta_grad,
                ad_eval_n_eta_converged=ad_eval_n_conv,
                wall_sec=wall,
                cpu_sec=cpu,
                method_eval_count=cache.evals,
                max_eta_grad_norm=cache.max_eta_grad,
                n_eta_converged=cache.n_converged,
                theta=theta_hat,
            )
            push!(rows, row)
            write_results(outpath, rows)
            @printf("  success=%s objective=%.6f ad_eval=%.6f wall=%.3f cpu=%.3f evals=%d\n",
                    string(row.success), row.objective, row.objective_ad_eval,
                    row.wall_sec, row.cpu_sec, row.method_eval_count)
        end
    end
    println("\nSaved ", length(rows), " rows to ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
