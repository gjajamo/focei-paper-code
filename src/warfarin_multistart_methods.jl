include("warfarin_forward_reverse_inner_compare.jl")

using Optim
using Random
using Base.Threads

const PARAM_NAMES = vcat([
    "logKA", "logCL", "logV",
    "logE0", "logC50", "logKE0", "logitEMAX",
    "logSIGMA_PK", "logSIGMA_PD",
], warfarin_combined_error() ? [
    "logSIGMA_PROP_PK", "logSIGMA_PROP_PD",
] : String[], [
    "logOM_KA", "logOM_CL", "logOM_V",
    "logOM_E0", "logOM_C50", "logOM_KE0",
])

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

primal_vector(v) = [primal_float(x) for x in v]
is_ad_active(v) = any(x -> x isa ForwardDiff.Dual || x isa ReverseDiff.TrackedReal, v)

function solve_mode_hessian(H, rhs)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    any(!isfinite, Hs) && error("non-finite exact mode Hessian in implicit solve")
    any(!isfinite, rhs) && error("non-finite right-hand side in implicit solve")
    F = cholesky(Symmetric(Hs); check=true)
    return F \ Vector{Float64}(rhs)
end

function shifted_matrix_ad(H; jitter::Float64=1.0e-4)
    q = size(H, 1)
    T = eltype(H)
    H_float = Matrix{Float64}(undef, q, q)
    for i in 1:q, j in 1:q
        H_float[i, j] = primal_float((H[i, j] + H[j, i]) / 2)
    end
    if any(!isfinite, H_float)
        error("non-finite Hessian in shifted_matrix_ad")
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
    step = small_spd_solve_ad(A, T.(g))

    nrm = sqrt(sum(abs2, primal_vector(step)))
    if !isfinite(nrm)
        error("non-finite unrolled Newton step")
    end
    if nrm > max_step_norm
        step = step .* (max_step_norm / (sqrt(sum(abs2, step)) + 1.0e-12))
    end
    return step
end

function small_spd_solve_ad(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    q = length(b)
    L = Matrix{T}(undef, q, q)
    for i in 1:q, j in 1:q
        L[i, j] = zero(T)
    end

    for i in 1:q
        for j in 1:i
            s = zero(T)
            for k in 1:(j - 1)
                s += L[i, k] * L[j, k]
            end
            if i == j
                v = A[i, i] - s
                if primal_float(v) <= 0.0 || !isfinite(primal_float(v))
                    error("non-positive pivot in differentiated SPD solve")
                end
                L[i, j] = sqrt(v)
            else
                L[i, j] = (A[i, j] - s) / L[j, j]
            end
        end
    end

    y = Vector{T}(undef, q)
    for i in 1:q
        s = zero(T)
        for k in 1:(i - 1)
            s += L[i, k] * y[k]
        end
        y[i] = (b[i] - s) / L[i, i]
    end

    x = Vector{T}(undef, q)
    for i in q:-1:1
        s = zero(T)
        for k in (i + 1):q
            s += L[k, i] * x[k]
        end
        x[i] = (y[i] - s) / L[i, i]
    end
    return x
end

mutable struct EtaWarmCache
    etas::Dict{Int, Vector{Float64}}
    best_f::Float64
end

EtaWarmCache() = EtaWarmCache(Dict{Int, Vector{Float64}}(), Inf)

function copy_eta_cache(cache::EtaWarmCache)
    return EtaWarmCache(Dict(k => copy(v) for (k, v) in cache.etas), cache.best_f)
end

function eta_start(cache::Union{Nothing,EtaWarmCache}, subj::SubjectData)
    if cache !== nothing && haskey(cache.etas, subj.sid)
        return copy(cache.etas[subj.sid])
    end
    return zeros(ETA_DIM)
end

function maybe_update_eta_cache!(cache::Union{Nothing,EtaWarmCache},
                                 subjects::Vector{SubjectData}, etas, f::Float64, g)
    cache === nothing && return
    isfinite(f) || return
    any(!isfinite, g) && return
    f <= cache.best_f || return
    for i in eachindex(subjects)
        eta = Vector{Float64}(etas[i])
        all(isfinite, eta) || return
    end
    for i in eachindex(subjects)
        cache.etas[subjects[i].sid] = copy(Vector{Float64}(etas[i]))
    end
    cache.best_f = f
end

function logdet_cholesky_ad(H; jitter::Float64=1.0e-4)
    mode = lowercase(get(ENV, "WARFARIN_JULIA_LOGDET_MODE", "raw"))
    regularized = mode in ("regularized", "shifted", "legacy")
    if mode == "raw"
        T = eltype(H)
        q = size(H, 1)
        A = Matrix{T}(undef, q, q)
        for i in 1:q, j in 1:q
            A[i, j] = (H[i, j] + H[j, i]) / 2
        end
    elseif regularized
        A = shifted_matrix_ad(H; jitter=jitter)
        T = eltype(A)
        q = size(A, 1)
    else
        error("Unknown WARFARIN_JULIA_LOGDET_MODE=$(mode); use raw or regularized")
    end

    L = zeros(T, q, q)
    for i in 1:q
        for j in 1:i
            s = A[i, j]
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end
            if i == j
                if primal_float(s) <= 0.0 || !isfinite(primal_float(s))
                    regularized || error("non-positive Cholesky pivot in raw FOCEI curvature")
                    s += T(abs(primal_float(s)) + jitter)
                end
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
        end
    end
    return 2 * sum(log(L[i, i]) for i in 1:q)
end

# Expected-information FOCEI curvature. For combined error, the second term
# retains the eta-dependent residual-variance sensitivity.
function focei_curvature(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    sigma_add_pk = exp(x[8])
    sigma_add_pd = exp(x[9])
    combined = warfarin_combined_error()
    sigma_prop_pk = combined ? exp(x[10]) : zero(sigma_add_pk)
    sigma_prop_pd = combined ? exp(x[11]) : zero(sigma_add_pd)
    omega_start = combined ? 12 : 10
    omega = exp.(x[omega_start:(omega_start + ETA_DIM - 1)])
    pk = predict_pk(subj, x, eta)
    pd = predict_pd(subj, x, eta, representation; dt=dt)
    Jpk = ForwardDiff.jacobian(e -> predict_pk(subj, x, e), eta)
    Jpd = ForwardDiff.jacobian(e -> predict_pd(subj, x, e, representation; dt=dt), eta)
    vpk = warfarin_variance.(pk, sigma_add_pk, sigma_prop_pk)
    vpd = warfarin_variance.(pd, sigma_add_pd, sigma_prop_pd)
    VJpk = ForwardDiff.jacobian(
        e -> warfarin_variance.(predict_pk(subj, x, e), sigma_add_pk, sigma_prop_pk), eta)
    VJpd = ForwardDiff.jacobian(
        e -> warfarin_variance.(predict_pd(subj, x, e, representation; dt=dt),
                                sigma_add_pd, sigma_prop_pd), eta)
    T = promote_type(eltype(Jpk), eltype(Jpd))
    G = transpose(Jpk) * Diagonal(inv.(vpk)) * Jpk +
        transpose(Jpd) * Diagonal(inv.(vpd)) * Jpd +
        0.5 .* transpose(VJpk) * Diagonal(inv.(vpk .* vpk)) * VJpk +
        0.5 .* transpose(VJpd) * Diagonal(inv.(vpd .* vpd)) * VJpd
    for j in 1:ETA_DIM
        G[j, j] += one(T) / (omega[j] * omega[j])
    end
    return G
end
function eta_grad_hess_ad(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    return ForwardDiff.gradient(f, eta), focei_curvature(subj, x, eta, representation; dt=dt), f(eta)
end

fd_rel_step(base::Float64, z::Float64) = base * max(1.0, abs(z))

function focei_curvature_fd(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                            representation::Symbol; dt::Float64=0.25, h::Float64=1.0e-3)
    q = length(eta)
    Jpk = zeros(length(subj.pk_obs), q)
    Jpd = zeros(length(subj.pd_obs), q)
    VJpk = zeros(length(subj.pk_obs), q)
    VJpd = zeros(length(subj.pd_obs), q)
    sigma_add_pk = exp(x[8])
    sigma_add_pd = exp(x[9])
    combined = warfarin_combined_error()
    sigma_prop_pk = combined ? exp(x[10]) : 0.0
    sigma_prop_pd = combined ? exp(x[11]) : 0.0
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        pkp = predict_pk(subj, x, ep)
        pkm = predict_pk(subj, x, em)
        pdp = predict_pd(subj, x, ep, representation; dt=dt)
        pdm = predict_pd(subj, x, em, representation; dt=dt)
        Jpk[:, j] .= (pkp .- pkm) ./ (2.0 * hj)
        Jpd[:, j] .= (pdp .- pdm) ./ (2.0 * hj)
        VJpk[:, j] .= (warfarin_variance.(pkp, sigma_add_pk, sigma_prop_pk) .-
                         warfarin_variance.(pkm, sigma_add_pk, sigma_prop_pk)) ./ (2.0 * hj)
        VJpd[:, j] .= (warfarin_variance.(pdp, sigma_add_pd, sigma_prop_pd) .-
                         warfarin_variance.(pdm, sigma_add_pd, sigma_prop_pd)) ./ (2.0 * hj)
    end

    pk = predict_pk(subj, x, eta)
    pd = predict_pd(subj, x, eta, representation; dt=dt)
    vpk = warfarin_variance.(pk, sigma_add_pk, sigma_prop_pk)
    vpd = warfarin_variance.(pd, sigma_add_pd, sigma_prop_pd)
    omega_start = combined ? 12 : 10
    omega = exp.(x[omega_start:(omega_start + q - 1)])
    G = transpose(Jpk) * Diagonal(inv.(vpk)) * Jpk +
        transpose(Jpd) * Diagonal(inv.(vpd)) * Jpd +
        0.5 .* transpose(VJpk) * Diagonal(inv.(vpk .* vpk)) * VJpk +
        0.5 .* transpose(VJpd) * Diagonal(inv.(vpd .* vpd)) * VJpd
    for j in 1:q
        G[j, j] += 1.0 / (omega[j] * omega[j])
    end
    return G
end

function fd_score_eta_central(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                      representation::Symbol; dt::Float64=0.25, h::Float64=1.0e-3)
    q = length(eta)
    g = zeros(q)
    for j in 1:q
        hj = fd_rel_step(h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        g[j] = (h_i(subj, x, ep, representation; dt=dt) -
                h_i(subj, x, em, representation; dt=dt)) / (2.0 * hj)
    end
    return g
end

function fd_score_eta(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                      representation::Symbol; dt::Float64=0.25, h::Float64=1.0e-3)
    g_h = fd_score_eta_central(subj, x, eta, representation; dt=dt, h=h)
    g_h2 = fd_score_eta_central(subj, x, eta, representation; dt=dt, h=0.5 * h)
    return (4.0 .* g_h2 .- g_h) ./ 3.0
end
function fd_grad_curvature_eta(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                               representation::Symbol; dt::Float64=0.25, h::Float64=1.0e-3)
    f0 = h_i(subj, x, eta, representation; dt=dt)
    g = fd_score_eta(subj, x, eta, representation; dt=dt, h=h)
    return g, focei_curvature_fd(subj, x, eta, representation; dt=dt, h=h), f0
end

function fd_exact_hessian_eta(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                              representation::Symbol; dt::Float64=0.25,
                              score_h::Float64=1.0e-3, hess_h::Float64=1.0e-3)
    q = length(eta)
    H = zeros(q, q)
    for j in 1:q
        hj = fd_rel_step(hess_h, eta[j])
        ep = copy(eta)
        em = copy(eta)
        ep[j] += hj
        em[j] -= hj
        gp = fd_score_eta(subj, x, ep, representation; dt=dt, h=score_h)
        gm = fd_score_eta(subj, x, em, representation; dt=dt, h=score_h)
        H[:, j] .= (gp .- gm) ./ (2.0 * hj)
    end
    return 0.5 .* (H .+ transpose(H))
end
function fd_newton_step(H, g; jitter::Float64=1.0e-6, floor_rel::Float64=1.0e-2,
                        floor_abs::Float64=1.0e-3, max_step_norm::Float64=5.0)
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

function strict_descent_accept(f_try::Float64, f0::Float64, g::Vector{Float64},
                               step::Vector{Float64}, alpha::Float64;
                               armijo::Float64=1.0e-4)
    isfinite(f_try) || return false
    predicted = alpha * dot(g, step)
    return predicted > 0.0 ? f_try <= f0 - armijo * predicted : f_try < f0
end

function exact_mode_polish_ad(subj::SubjectData, x::Vector{Float64}, eta0::Vector{Float64},
                              representation::Symbol; dt::Float64=0.25,
                              maxiter::Int=8, tol::Float64=1.0e-7)
    eta = copy(eta0)
    for _ in 1:maxiter
        f = e -> h_i(subj, x, e, representation; dt=dt)
        f0 = Float64(f(eta))
        g = Vector{Float64}(ForwardDiff.gradient(f, eta))
        isfinite(f0) && all(isfinite, g) || break
        norm(g) < tol && break
        H = Matrix{Float64}(ForwardDiff.hessian(f, eta))
        all(isfinite, H) || break
        step = try
            solve_mode_hessian(H, g)
        catch
            safe_solve(H, g; jitter=1.0e-10)
        end
        all(isfinite, step) && norm(step) > 0.0 || break
        alpha = 1.0
        accepted = false
        for _ls in 1:20
            trial = eta .- alpha .* step
            f_try = Float64(f(trial))
            trial_score = Vector{Float64}(ForwardDiff.gradient(f, trial))
            roundoff_band = 64.0 * eps(Float64) * max(1.0, abs(f0))
            score_decreased = norm(trial_score) <
                              (1.0 - 1.0e-4 * alpha) * norm(g)
            if strict_descent_accept(f_try, f0, g, step, alpha) ||
               (f_try <= f0 + roundoff_band && score_decreased)
                eta = trial
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    final_score = ForwardDiff.gradient(e -> h_i(subj, x, e, representation; dt=dt), eta)
    grad_norm = norm(final_score)
    return eta, grad_norm, isfinite(grad_norm) && grad_norm < tol
end

function exact_mode_polish_fd(subj::SubjectData, x::Vector{Float64}, eta0::Vector{Float64},
                              representation::Symbol; dt::Float64=0.25,
                              maxiter::Int=8, tol::Float64=1.0e-5,
                              eps_eta::Float64=1.0e-4)
    eta = copy(eta0)
    for _ in 1:maxiter
        f0 = Float64(h_i(subj, x, eta, representation; dt=dt))
        g = fd_score_eta(subj, x, eta, representation; dt=dt, h=eps_eta)
        isfinite(f0) && all(isfinite, g) || break
        norm(g) < tol && break
        H = fd_exact_hessian_eta(subj, x, eta, representation;
                                 dt=dt, score_h=eps_eta, hess_h=eps_eta)
        all(isfinite, H) || break
        step = fd_newton_step(H, g; jitter=1.0e-10, floor_rel=0.0,
                              floor_abs=0.0, max_step_norm=3.0)
        all(isfinite, step) && norm(step) > 0.0 || break
        alpha = 1.0
        accepted = false
        for _ls in 1:20
            trial = eta .- alpha .* step
            f_try = Float64(h_i(subj, x, trial, representation; dt=dt))
            trial_score = fd_score_eta(
                subj, x, trial, representation; dt=dt, h=eps_eta
            )
            roundoff_band = 64.0 * eps(Float64) * max(1.0, abs(f0))
            score_decreased = norm(trial_score) <
                              (1.0 - 1.0e-4 * alpha) * norm(g)
            if strict_descent_accept(f_try, f0, g, step, alpha) ||
               (f_try <= f0 + roundoff_band && score_decreased)
                eta = trial
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    final_score = fd_score_eta(subj, x, eta, representation; dt=dt, h=eps_eta)
    grad_norm = norm(final_score)
    return eta, grad_norm, isfinite(grad_norm) && grad_norm < tol
end
function eta_mode_newton_focei(subj::SubjectData, x::Vector{Float64}, representation::Symbol;
                               dt::Float64=0.25, maxiter::Int=30, tol::Float64=1.0e-7,
                               eta0::AbstractVector{<:Real}=zeros(ETA_DIM))
    eta = Float64.(eta0)
    for _ in 1:maxiter
        g, G, f0 = eta_grad_hess_ad(subj, x, eta, representation; dt=dt)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, G)
            break
        end
        norm(g) < tol && break
        step = safe_solve(G, g)
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial = eta .- alpha .* step
            f_try = Float64(h_i(subj, x, trial, representation; dt=dt))
            if strict_descent_accept(f_try, Float64(f0), Vector{Float64}(g),
                                     Vector{Float64}(step), alpha)
                eta = trial
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end

    final_score = ForwardDiff.gradient(e -> h_i(subj, x, e, representation; dt=dt), eta)
    grad_norm = norm(final_score)
    converged = isfinite(grad_norm) && grad_norm < tol
    polish_maxiter = parse(Int, get(ENV, "WARFARIN_JULIA_ETA_POLISH_MAXITER", "8"))
    if !converged && polish_maxiter > 0
        eta, grad_norm, converged = exact_mode_polish_ad(
            subj, x, eta, representation; dt=dt, maxiter=polish_maxiter, tol=tol
        )
    end
    return eta, Float64(h_i(subj, x, eta, representation; dt=dt)), grad_norm, converged
end

function eta_mode_newton_fd(subj::SubjectData, x::Vector{Float64}, representation::Symbol;
                            dt::Float64=0.25, maxiter::Int=30, tol::Float64=1.0e-7,
                            eps_eta::Float64=1.0e-4,
                            floor_rel::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_FLOOR_REL", "1e-6")),
                            floor_abs::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_FLOOR_ABS", "1e-3")),
                            max_step_norm::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_ETA_MAX_STEP", "5.0")),
                            eta0::AbstractVector{<:Real}=zeros(ETA_DIM))
    eta = Float64.(eta0)
    f_curr = h_i(subj, x, eta, representation; dt=dt)
    grad_norm = Inf
    converged = false
    for _ in 1:maxiter
        g, H, f0 = fd_grad_curvature_eta(subj, x, eta, representation; dt=dt, h=eps_eta)
        grad_norm = norm(g)
        if !isfinite(f0) || any(!isfinite, g) || any(!isfinite, H)
            break
        end
        if grad_norm < tol
            f_curr = f0
            converged = true
            break
        end
        step = fd_newton_step(H, g; floor_rel=floor_rel, floor_abs=floor_abs,
                              max_step_norm=max_step_norm)
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial = eta .- alpha .* step
            f_try = Float64(h_i(subj, x, trial, representation; dt=dt))
            if strict_descent_accept(f_try, Float64(f0), Vector{Float64}(g),
                                     Vector{Float64}(step), alpha)
                eta = trial
                f_curr = f_try
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    final_score = fd_score_eta(subj, x, eta, representation; dt=dt, h=eps_eta)
    grad_norm = norm(final_score)
    convergence_tol = tol
    converged = isfinite(grad_norm) && grad_norm < convergence_tol
    polish_maxiter = parse(Int, get(ENV, "WARFARIN_JULIA_FD_ETA_POLISH_MAXITER", "8"))
    if !converged && polish_maxiter > 0
        eta, grad_norm, converged = exact_mode_polish_fd(
            subj, x, eta, representation; dt=dt, maxiter=polish_maxiter,
            tol=convergence_tol, eps_eta=eps_eta
        )
    end
    return eta, Float64(h_i(subj, x, eta, representation; dt=dt)), grad_norm, converged
end

function solve_all_etas(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                        dt::Float64=0.25, maxiter::Int=30, backend::Symbol=:forward,
                        eps_eta::Float64=1.0e-4,
                        eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    # BitVector writes are not independent across threads because flags share
    # packed machine words.
    converged = Vector{Bool}(undef, length(subjects))
    @threads for i in eachindex(subjects)
        eta0 = eta_start(eta_cache, subjects[i])
        if backend == :fd
            eta, _, gn, cvg = eta_mode_newton_fd(subjects[i], x, representation; dt=dt,
                                                 maxiter=maxiter, eps_eta=eps_eta, eta0=eta0)
        else
            eta, _, gn, cvg = eta_mode_newton_focei(subjects[i], x, representation;
                                                    dt=dt, maxiter=maxiter, eta0=eta0)
        end
        etas[i] = eta
        grad_norms[i] = gn
        converged[i] = cvg
    end
    return etas, maximum(grad_norms), count(converged)
end

function focei_subject_fixed_eta(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    f = e -> h_i(subj, x, e, representation; dt=dt)
    H = focei_curvature(subj, x, eta, representation; dt=dt)
    hi = f(eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky_ad(H))
end

function focei_subject_fixed_eta_fd(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                     representation::Symbol; dt::Float64=0.25,
                                     eps_eta::Float64=1.0e-4)
    hi = h_i(subj, x, eta, representation; dt=dt)
    ld = strict_fd_logdet(focei_curvature_fd(subj, x, eta, representation; dt=dt, h=eps_eta))
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

function outer_gradient(f, x::Vector{Float64})
    mode = lowercase(get(ENV, "WARFARIN_JULIA_OUTER_AD_MODE", "forward"))
    if mode in ("reverse", "backward")
        return ReverseDiff.gradient(f, x)
    elseif mode == "forward"
        return ForwardDiff.gradient(f, x)
    end
    error("Unknown WARFARIN_JULIA_OUTER_AD_MODE=$(mode); use reverse or forward")
end

function stop_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                         dt::Float64=0.25, maxiter_eta::Int=30,
                         eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta = etas[i]
        fx = xx -> focei_subject_fixed_eta(subj, xx, eta, representation; dt=dt)
        vals[i] = Float64(fx(x))
        grads[i] = outer_gradient(fx, x)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function ad_population_value(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                             dt::Float64=0.25, maxiter_eta::Int=30)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt, maxiter=maxiter_eta, backend=:forward)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = Float64(focei_subject_fixed_eta(subjects[i], x, etas[i], representation; dt=dt))
    end
    return sum(vals), max_eta_grad, n_conv
end

function safe_ad_population_evaluation(subjects::Vector{SubjectData}, x::Vector{Float64},
                                       representation::Symbol; dt::Float64=0.25,
                                       maxiter_eta::Int=30)
    try
        return ad_population_value(subjects, x, representation; dt=dt, maxiter_eta=maxiter_eta)
    catch err
        @warn "AD endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN, NaN, 0
    end
end

safe_ad_population_value(subjects::Vector{SubjectData}, x::Vector{Float64},
                         representation::Symbol; dt::Float64=0.25,
                         maxiter_eta::Int=30) =
    safe_ad_population_evaluation(subjects, x, representation; dt=dt,
                                  maxiter_eta=maxiter_eta)[1]

function full_implicit_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                          representation::Symbol; dt::Float64=0.25)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    ofv = focei_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    dh_dx = ForwardDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(
        xx -> logdet_cholesky_ad(focei_curvature(subj, xx, eta, representation; dt=dt)),
        x,
    )
    dld_deta = ForwardDiff.gradient(
        ee -> logdet_cholesky_ad(focei_curvature(subj, x, ee, representation; dt=dt)),
        eta,
    )
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(
        xx -> dot(ForwardDiff.gradient(e -> h_i(subj, xx, e, representation; dt=dt), eta), lambda),
        x,
    )
    grad = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- term_c)
    return Float64(ofv), Vector{Float64}(grad)
end

function full_implicit_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                  dt::Float64=0.25, maxiter_eta::Int=30,
                                  eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        vals[i], grads[i] = full_implicit_subject_value_grad(subjects[i], x, etas[i], representation; dt=dt)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

# Solve the primal EBE to convergence, detach it, and differentiate one exact
# Newton update. At an exact root this has the same local mode sensitivity as
# implicit differentiation, but it records a distinct one-step graph.
function one_step_newton_subject_value(subj::SubjectData, x, eta_star::Vector{Float64},
                                       representation::Symbol; dt::Float64=0.25)
    eta = eltype(x).(eta_star)
    g = ForwardDiff.gradient(e -> h_i(subj, x, e, representation; dt=dt), eta)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    step = small_spd_solve_ad(H, g)
    return focei_subject_fixed_eta(subj, x, eta .- step, representation; dt=dt)
end

function one_step_newton_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64},
                                    representation::Symbol; dt::Float64=0.25,
                                    maxiter_eta::Int=30,
                                    eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    n_conv == length(subjects) || error("one-step Newton requires converged primal EBEs")
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta_star = etas[i]
        fx = xx -> one_step_newton_subject_value(subj, xx, eta_star, representation; dt=dt)
        vals[i] = Float64(fx(x))
        grads[i] = outer_gradient(fx, x)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, max_eta_grad, n_conv
end

function eta_newton_unroll(subj::SubjectData, x, eta0::Vector{Float64}, representation::Symbol;
                           dt::Float64=0.25, steps::Int=30, tol::Float64=1.0e-6)
    T = eltype(x)
    eta = T.(eta0)
    for _ in 1:steps
        g, H, hi = eta_grad_hess_ad(subj, x, eta, representation; dt=dt)
        if !isfinite(primal_float(hi)) || any(!isfinite, primal_vector(g))
            error("non-finite unrolled eta gradient")
        end
        step = unroll_newton_step(H, g)
        step_norm = sqrt(sum(abs2, primal_vector(step)))

        # Match the Python implementation: line-search decisions are made on
        # detached/primal values, then the accepted scalar alpha is applied to
        # the differentiable Newton update.
        x_pr = primal_vector(x)
        eta_pr = primal_vector(eta)
        step_pr = primal_vector(step)
        f0 = h_i(subj, x_pr, eta_pr, representation; dt=dt)
        if !isfinite(Float64(f0))
            error("non-finite unrolled eta objective")
        end
        alpha = 1.0
        accepted = false
        for _ls in 1:14
            trial_pr = eta_pr .- alpha .* step_pr
            f_try = h_i(subj, x_pr, trial_pr, representation; dt=dt)
            if isfinite(Float64(f_try)) && Float64(f_try) <= Float64(f0) + 1.0e-10
                eta = eta .- alpha .* step
                accepted = true
                break
            end
            alpha *= 0.5
        end
        accepted || break
    end
    return eta
end

function full_unroll_subject_value(subj::SubjectData, x, eta0::Vector{Float64}, representation::Symbol;
                                   dt::Float64=0.25, steps::Int=30)
    eta = eta_newton_unroll(subj, x, eta0, representation; dt=dt, steps=steps)
    return focei_subject_fixed_eta(subj, x, eta, representation; dt=dt)
end

function full_unroll_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                dt::Float64=0.25, steps::Int=30,
                                eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    vals = zeros(length(subjects))
    grads = [zeros(length(x)) for _ in subjects]
    etas = Vector{Vector{Float64}}(undef, length(subjects))
    grad_norms = zeros(length(subjects))
    diagnostic_tol = parse(Float64, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_ETA_TOL", "1e-6"))
    @threads for i in eachindex(subjects)
        subj = subjects[i]
        eta0 = eta_start(eta_cache, subj)
        eta = eta_newton_unroll(subj, x, eta0, representation; dt=dt, steps=steps)
        etas[i] = primal_vector(eta)
        final_score = ForwardDiff.gradient(e -> h_i(subj, x, e, representation; dt=dt), etas[i])
        grad_norms[i] = norm(final_score)
        vals[i] = Float64(focei_subject_fixed_eta(subj, x, eta, representation; dt=dt))
        fx = xx -> full_unroll_subject_value(subj, xx, eta0, representation; dt=dt, steps=steps)
        grads[i] = outer_gradient(fx, x)
    end
    total = sum(vals)
    total_grad = vec(sum(reduce(hcat, grads), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_grad)
    return total, total_grad, maximum(grad_norms), count(<(diagnostic_tol), grad_norms)
end

function fd_population_value(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                             dt::Float64=0.25, maxiter_eta::Int=30,
                             eps_eta::Float64=1.0e-4,
                             eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:fd,
                                                eps_eta=eps_eta, eta_cache=eta_cache)
    require_convergence = lowercase(get(ENV, "WARFARIN_JULIA_FD_REQUIRE_PERTURBED_ETA_CONVERGENCE", "true")) in
                          ("1", "true", "yes", "on")
    if require_convergence && n_conv != length(subjects)
        error("FD perturbed EBE solve converged for $n_conv/$(length(subjects)); max score norm=$max_eta_grad")
    end
    return fd_population_value_from_etas(subjects, x, etas, representation; dt=dt,
                                         eps_eta=eps_eta)
end

function fd_population_value_from_etas(subjects::Vector{SubjectData}, x::Vector{Float64}, etas,
                                       representation::Symbol; dt::Float64=0.25,
                                       eps_eta::Float64=1.0e-4)
    vals = zeros(length(subjects))
    @threads for i in eachindex(subjects)
        vals[i] = focei_subject_fixed_eta_fd(subjects[i], x, etas[i], representation;
                                               dt=dt, eps_eta=eps_eta)
    end
    return sum(vals)
end

function fd_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                       dt::Float64=0.25, maxiter_eta::Int=30,
                       eps::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_EPS_THETA", "1e-3")),
                       eps_eta::Float64=parse(Float64, get(ENV, "WARFARIN_JULIA_FD_EPS_ETA", "1e-4")),
                       eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:fd,
                                                eps_eta=eps_eta, eta_cache=eta_cache)
    f0 = fd_population_value_from_etas(subjects, x, etas, representation; dt=dt,
                                       eps_eta=eps_eta)
    base_cache = EtaWarmCache()
    for i in eachindex(subjects)
        base_cache.etas[subjects[i].sid] = copy(etas[i])
    end
    base_cache.best_f = isfinite(f0) ? f0 : Inf
    g = zeros(length(x))
    lo, hi = theta_bounds()
    @threads for j in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        h = eps * max(1.0, abs(x[j]))
        xp[j] = min(max(x[j] + h, lo[j]), hi[j])
        xm[j] = min(max(x[j] - h, lo[j]), hi[j])
        step_p = xp[j] - x[j]
        step_m = x[j] - xm[j]
        if step_p == 0.0 && step_m == 0.0
            g[j] = 0.0
        elseif step_p == 0.0 || step_m == 0.0
            xone = step_p == 0.0 ? xm : xp
            step = step_p == 0.0 ? -step_m : step_p
            f1 = fd_population_value(subjects, xone, representation; dt=dt,
                                      maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                      eta_cache=copy_eta_cache(base_cache))
            isfinite(f1) || error("non-finite one-sided FD population objective for parameter $j")
            g[j] = (f1 - f0) / step
        else
            fp = fd_population_value(subjects, xp, representation; dt=dt,
                                     maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_cache=copy_eta_cache(base_cache))
            fm = fd_population_value(subjects, xm, representation; dt=dt,
                                     maxiter_eta=maxiter_eta, eps_eta=eps_eta,
                                     eta_cache=copy_eta_cache(base_cache))
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
    fd_cache_tol = parse(Float64, get(ENV, "WARFARIN_JULIA_FD_CACHE_MAX_ETA_GRAD", "1e-3"))
    if n_conv == length(subjects) || (isfinite(max_eta_grad) && max_eta_grad <= fd_cache_tol)
        maybe_update_eta_cache!(eta_cache, subjects, etas, Float64(f0), g)
    end
    return f0, g, max_eta_grad, n_conv
end

function method_evaluator(method::String, subjects::Vector{SubjectData}, representation::Symbol;
                          dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30)
    eta_cache = EtaWarmCache()
    if method == "STOP"
        return x -> stop_value_grad(subjects, x, representation; dt=dt,
                                    maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT"
        return x -> full_implicit_value_grad(subjects, x, representation; dt=dt,
                                             maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT_REVERSE_VJP"
        return x -> full_implicit_reverse_vjp_value_grad(subjects, x, representation; dt=dt,
                                                         maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "HYBRID_FF_REVERSE_VJP"
        return x -> full_implicit_hybrid_ff_reverse_vjp_value_grad(subjects, x, representation; dt=dt,
                                                                     maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "HYBRID_RF_REVERSE_VJP"
        return x -> full_implicit_hybrid_rf_reverse_vjp_value_grad(subjects, x, representation; dt=dt,
                                                                     maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "HYBRID_RRH_REVERSE_VJP"
        return x -> full_implicit_hybrid_rrh_reverse_vjp_value_grad(subjects, x, representation; dt=dt,
                                                                      maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "REVERSE_DIRECT_FORWARD_CONTRACTION"
        return x -> full_implicit_reverse_direct_forward_contraction_value_grad(subjects, x, representation; dt=dt,
                                                                                maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_IMPLICIT_DIRECTIONAL_JVP"
        return x -> full_implicit_directional_jvp_value_grad(subjects, x, representation; dt=dt,
                                                           maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "ALMQUIST_FORWARD"
        return x -> almquist_forward_value_grad(subjects, x, representation; dt=dt,
                                                maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "LAPLACE_IMPLICIT"
        return x -> laplace_value_grad(subjects, x, representation; dt=dt,
                                       maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FULL_UNROLL"
        use_unroll_cache = lowercase(get(ENV, "WARFARIN_JULIA_FULL_UNROLL_USE_ETA_CACHE", "false")) in
                           ("1", "true", "yes", "on")
        unroll_cache = use_unroll_cache ? eta_cache : nothing
        return x -> full_unroll_value_grad(subjects, x, representation; dt=dt,
                                           steps=full_unroll_steps, eta_cache=unroll_cache)
    elseif method == "FULL_UNROLL_1NEWTON"
        return x -> one_step_newton_value_grad(subjects, x, representation; dt=dt,
                                               maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    elseif method == "FD"
        return x -> fd_value_grad(subjects, x, representation; dt=dt,
                                  maxiter_eta=maxiter_eta, eta_cache=eta_cache)
    end
    error("unknown method: $method")
end

function canonical_method(method::String)
    m = uppercase(strip(method))
    m == "FULL_IMPLIC" && return "FULL_IMPLICIT"
    m == "STOP+FULL_IMPLIC" && return "STOP+FULL_IMPLICIT"
    m == "FULL_IMPLIC+STOP" && return "FULL_IMPLICIT+STOP"
    return m
end

function stage_plan(method::String, maxiter_outer::Int)
    m = canonical_method(method)
    if m in ("STOP+FULL", "STOP+FULL_IMPLICIT")
        stage1 = parse(Int, get(ENV, "WARFARIN_JULIA_STOP_FULL_STOP_ITERS", "1"))
        stage1 = max(1, min(stage1, maxiter_outer - 1))
        stage2 = parse(Int, get(ENV, "WARFARIN_JULIA_STOP_FULL_FULL_ITERS", string(maxiter_outer - stage1)))
        return "STOP+FULL_IMPLICIT", "STOP", "FULL_IMPLICIT", stage1, max(1, stage2)
    elseif m in ("FULL+STOP", "FULL_IMPLICIT+STOP")
        default_full = string(max(1, min(14, maxiter_outer - 1)))
        stage1 = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_STOP_FULL_ITERS", default_full))
        stage1 = max(1, min(stage1, maxiter_outer - 1))
        stage2 = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_STOP_STOP_ITERS", string(maxiter_outer - stage1)))
        return "FULL_IMPLICIT+STOP", "FULL_IMPLICIT", "STOP", stage1, max(1, stage2)
    end
    return m, "", "", 0, 0
end

is_staged_method(method::String) = stage_plan(method, 50)[2] != ""

function method_env_key(method::String, suffix::String)
    clean = replace(uppercase(strip(method)), r"[^A-Z0-9]" => "_")
    return "WARFARIN_JULIA_$(clean)_$(suffix)"
end

function base_x0()
    fixed = [
        log(1.0), log(0.2), log(10.0),
        log(100.0), log(2.0), log(0.05), log(0.9 / 0.1),
        log(0.5), log(5.0),
    ]
    residual_extra = warfarin_combined_error() ? [log(0.1), log(0.1)] : Float64[]
    return vcat(fixed, residual_extra, [
        fill(log(0.3), 6)...,
    ])
end

function theta_bounds()
    lo_fixed = [log(1.0e-3), log(1.0e-4), log(1.0e-2),
          log(1.0), log(1.0e-3), log(1.0e-4), -10.0,
          log(1.0e-4), log(1.0e-4)]
    hi_fixed = [log(50.0), log(10.0), log(1.0e3),
          log(200.0), log(1.0e3), log(10.0), 10.0,
          log(50.0), log(200.0)]
    lo_residual = warfarin_combined_error() ? [log(1.0e-4), log(1.0e-4)] : Float64[]
    hi_residual = warfarin_combined_error() ? [log(5.0), log(5.0)] : Float64[]
    lo = vcat(lo_fixed, lo_residual, fill(log(1.0e-4), 6))
    hi = vcat(hi_fixed, hi_residual, fill(log(5.0), 6))
    return lo, hi
end

function sample_starts(n::Int, base::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                       seed::Int=123579, scale::Float64=0.5)
    rng = MersenneTwister(seed)
    starts = Matrix{Float64}(undef, n, length(base))
    for s in 1:n
        starts[s, :] = min.(max.(base .+ randn(rng, length(base)) .* scale, lo), hi)
    end
    return starts
end

function write_start_bank(path::AbstractString, starts::Matrix{Float64})
    open(path, "w") do io
        println(io, join(vcat(["model", "start_id"], PARAM_NAMES), ","))
        for s in 1:size(starts, 1)
            println(io, join(vcat(["warfarin", string(s - 1)], string.(Vector{Float64}(starts[s, :]))), ","))
        end
    end
end

function read_start_bank_csv(path::AbstractString, lo::Vector{Float64}, hi::Vector{Float64}; n_starts::Int)
    lines = readlines(path)
    isempty(lines) && error("empty start bank: $path")
    header = strip.(split(lines[1], ','))
    indices = Int[]
    for name in PARAM_NAMES
        idx = findfirst(==(name), header)
        idx === nothing && error("start bank $path is missing parameter column $name")
        push!(indices, idx)
    end
    rows = Vector{Vector{Float64}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = strip.(split(line, ','))
        length(fields) >= maximum(indices) || error("malformed start-bank row in $path")
        theta = [parse(Float64, fields[idx]) for idx in indices]
        length(theta) == length(lo) || error("start bank dimensionality mismatch")
        theta .= min.(max.(theta, lo), hi)
        push!(rows, theta)
        length(rows) >= n_starts && break
    end
    length(rows) < n_starts && error("start bank $path has $(length(rows)) rows; expected at least $n_starts")
    starts = Matrix{Float64}(undef, n_starts, length(PARAM_NAMES))
    for i in 1:n_starts
        starts[i, :] .= rows[i]
    end
    return starts
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

EvalCache(p::Int) = EvalCache(false, zeros(p), Inf, zeros(p), NaN, 0, 0, false,
                              false, zeros(p), Inf)

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
            cache.f = BIG
            cache.g = zeros(length(xx))
        end
        cache.max_eta_grad = NaN
        cache.n_converged = 0
        cache.failed = true
        @warn "method evaluation failed; returning penalty" exception=typeof(err)
    end
    cache.valid = true
    cache.evals += 1
end

function optimize_one(method::String, subjects::Vector{SubjectData}, representation::Symbol,
                      theta0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                      dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30,
                      maxiter_outer::Int=50)
    evaluator = method_evaluator(method, subjects, representation; dt=dt, maxiter_eta=maxiter_eta,
                                 full_unroll_steps=full_unroll_steps)
    cache = EvalCache(length(theta0))
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
    f(x) = begin
        outside, fp, _ = bounds_penalty(x)
        outside && return fp
        ensure_cache!(cache, evaluator, x)
        cache.f
    end
    function g!(G, x)
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
    linesearch = lowercase(get(ENV, method_env_key(method, "LINESEARCH"),
                               get(ENV, "WARFARIN_JULIA_LINESEARCH", "hagerzhang")))
    method_obj = linesearch == "backtracking" ? LBFGS(linesearch=Optim.LineSearches.BackTracking()) : LBFGS()
    result = try
        optimize(f, g!, theta0, method_obj,
                 Optim.Options(iterations=maxiter_outer, g_tol=1.0e-6,
                               f_reltol=1.0e-8, show_trace=false))
    catch err
        @warn "outer optimizer failed; returning last finite point" method=method exception=typeof(err)
        fallback_x = cache.has_finite ? copy(cache.last_finite_x) : copy(theta0)
        cache.failed = true
        OptimFallbackResult(fallback_x, 0, false)
    end
    wall = (time_ns() - wall0) / 1.0e9
    cpu = process_cpu_seconds() - cpu0
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result)), lo), hi)
    ensure_cache!(cache, evaluator, theta_hat)
    return result, cache, wall, cpu
end

function optimize_method(method::String, subjects::Vector{SubjectData}, representation::Symbol,
                         theta0::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                         dt::Float64=0.25, maxiter_eta::Int=30, full_unroll_steps::Int=30,
                         maxiter_outer::Int=50)
    display_method, first_method, second_method, first_iters, second_iters = stage_plan(method, maxiter_outer)
    if isempty(first_method)
        result, cache, wall, cpu = optimize_one(display_method, subjects, representation, theta0, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=maxiter_outer)
        theta_hat = Vector{Float64}(Optim.minimizer(result))
        return (
            method=display_method,
            theta=theta_hat,
            iterations=Optim.iterations(result),
            converged=Optim.converged(result),
            objective=cache.f,
            wall=wall,
            cpu=cpu,
            evals=cache.evals,
            max_eta_grad=cache.max_eta_grad,
            n_converged=cache.n_converged,
            failed=cache.failed,
            stage1_method="",
            stage1_outer=0,
            stage2_method="",
            stage2_outer=0,
        )
    end

    result1, cache1, wall1, cpu1 = optimize_one(first_method, subjects, representation, theta0, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=first_iters)
    theta1 = min.(max.(Vector{Float64}(Optim.minimizer(result1)), lo), hi)
    result2, cache2, wall2, cpu2 = optimize_one(second_method, subjects, representation, theta1, lo, hi;
                                                dt=dt, maxiter_eta=maxiter_eta,
                                                full_unroll_steps=full_unroll_steps,
                                                maxiter_outer=second_iters)
    theta_hat = min.(max.(Vector{Float64}(Optim.minimizer(result2)), lo), hi)
    return (
        method=display_method,
        theta=theta_hat,
        iterations=Optim.iterations(result1) + Optim.iterations(result2),
        converged=Optim.converged(result2),
        objective=cache2.f,
        wall=wall1 + wall2,
        cpu=cpu1 + cpu2,
        evals=cache1.evals + cache2.evals,
        max_eta_grad=cache2.max_eta_grad,
        n_converged=cache2.n_converged,
        failed=cache1.failed || cache2.failed,
        stage1_method=first_method,
        stage1_outer=first_iters,
        stage2_method=second_method,
        stage2_outer=second_iters,
    )
end

function with_env_value(f::Function, key::String, value::String)
    had_key = haskey(ENV, key)
    old_value = had_key ? ENV[key] : ""
    ENV[key] = value
    try
        return f()
    finally
        if had_key
            ENV[key] = old_value
        else
            delete!(ENV, key)
        end
    end
end

function account_full_unroll_retry(primary, retry; use_retry::Bool)
    chosen = use_retry ? retry : primary
    return (
        method=chosen.method,
        theta=chosen.theta,
        iterations=primary.iterations + retry.iterations,
        converged=chosen.converged,
        objective=chosen.objective,
        wall=primary.wall + retry.wall,
        cpu=primary.cpu + retry.cpu,
        evals=primary.evals + retry.evals,
        max_eta_grad=chosen.max_eta_grad,
        n_converged=chosen.n_converged,
        failed=primary.failed || retry.failed,
        stage1_method="FULL_UNROLL_HAGERZHANG",
        stage1_outer=primary.iterations,
        stage2_method=use_retry ? "FULL_UNROLL_BACKTRACKING" : "FULL_UNROLL_BACKTRACKING_REJECTED",
        stage2_outer=retry.iterations,
    )
end

function write_results(path::AbstractString, rows)
    open(path, "w") do io
        header = vcat([
            "model", "implementation", "representation", "method", "start_id", "n_subjects",
            "maxiter_eta", "full_unroll_steps", "outer_iterations", "success", "objective",
            "objective_ad_eval", "objective_stop_eval", "ad_eval_max_eta_grad_norm",
            "ad_eval_n_eta_converged", "wall_sec", "cpu_sec", "method_eval_count",
            "max_eta_grad_norm", "n_eta_converged", "stage1_method", "stage1_outer",
            "stage2_method", "stage2_outer",
        ], PARAM_NAMES)
        println(io, join(header, ","))
        for r in rows
            vals = vcat([
                r.model, r.implementation, r.representation, r.method, r.start_id, r.n_subjects,
                r.maxiter_eta, r.full_unroll_steps, r.outer_iterations, r.success, r.objective,
                r.objective_ad_eval, r.objective_stop_eval, r.ad_eval_max_eta_grad_norm,
                r.ad_eval_n_eta_converged, r.wall_sec, r.cpu_sec, r.method_eval_count,
                r.max_eta_grad_norm, r.n_eta_converged, r.stage1_method, r.stage1_outer,
                r.stage2_method, r.stage2_outer,
            ], r.theta)
            println(io, join(string.(vals), ","))
        end
    end
end

split_tokens(s::String) = [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]

function parse_rep(s::String)
    ss = lowercase(strip(s))
    ss == "closed_form" && return :closed_form
    ss == "closed-form" && return :closed_form
    ss == "ode" && return :ode
    error("unknown representation: $s")
end

function main()
    root = normpath(joinpath(@__DIR__, ".."))
    data_path = get(ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "warfarin_dat.csv"))
    subjects_all = parse_warfarin_csv(data_path)
    n_subjects = parse(Int, get(ENV, "WARFARIN_JULIA_N_SUBJ", string(length(subjects_all))))
    subjects = subjects_all[1:min(n_subjects, length(subjects_all))]
    n_starts = parse(Int, get(ENV, "WARFARIN_JULIA_N_STARTS", "10"))
    maxiter_eta = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_ETA", "30"))
    maxiter_outer = parse(Int, get(ENV, "WARFARIN_JULIA_MAXITER_OUTER", "50"))
    full_unroll_steps = parse(Int, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_STEPS", string(maxiter_eta)))
    dt = parse(Float64, get(ENV, "WARFARIN_JULIA_DT", "0.25"))
    methods = split_tokens(get(ENV, "WARFARIN_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL,STOP,FD"))
    reps = [parse_rep(x) for x in split_tokens(get(ENV, "WARFARIN_JULIA_REPRESENTATIONS", "ode"))]
    outdir = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "tables"))
    mkpath(outdir)
    outpath = joinpath(outdir, "warfarin_julia_multistart_methods.csv")
    starts_path = joinpath(outdir, "warfarin_julia_start_bank.csv")
    full_unroll_auto_retry = lowercase(get(ENV, "WARFARIN_JULIA_FULL_UNROLL_AUTO_RETRY", "false")) in ("1", "true", "yes")
    full_unroll_retry_delta = parse(Float64, get(ENV, "WARFARIN_JULIA_FULL_UNROLL_RETRY_DELTA", "0.5"))

    lo, hi = theta_bounds()
    starts_csv = get(ENV, "WARFARIN_JULIA_STARTS_CSV", "")
    starts = isempty(starts_csv) ? sample_starts(n_starts, base_x0(), lo, hi) :
        read_start_bank_csv(starts_csv, lo, hi; n_starts=n_starts)
    write_start_bank(starts_path, starts)

    println("Warfarin Julia multistart methods")
    println("threads=", nthreads(), " subjects=", length(subjects), " starts=", n_starts)
    println("representations=", join(string.(reps), ","), " methods=", join(methods, ","))
    println("maxiter_eta=", maxiter_eta, " full_unroll_steps=", full_unroll_steps,
            " maxiter_outer=", maxiter_outer, " dt=", dt)
    println("output=", outpath)
    println("starts=", starts_path)
    !isempty(starts_csv) && println("matched_starts_csv=", starts_csv)

    for rep in reps, method in methods
        println("warming ", rep, " / ", method)
        _, first_method, second_method, _, _ = stage_plan(method, maxiter_outer)
        warm_methods = isempty(first_method) ? [canonical_method(method)] : [first_method, second_method]
        for warm_method in warm_methods
            evaluator = method_evaluator(warm_method, subjects, rep; dt=dt,
                                         maxiter_eta=min(maxiter_eta, 2),
                                         full_unroll_steps=min(full_unroll_steps, 2))
            try
                evaluator(Vector{Float64}(starts[1, :]))
            catch err
                @warn "warm-up evaluation failed; continuing to the configured production evaluator" representation=rep method=warm_method exception=typeof(err)
            end
        end
    end

    rows = NamedTuple[]
    for rep in reps
        for method in methods
            for s in 1:n_starts
                theta0 = Vector{Float64}(starts[s, :])
                @printf("\n[%s/%s] start %d/%d\n", string(rep), method, s, n_starts)
                outcome = optimize_method(method, subjects, rep, theta0, lo, hi;
                                          dt=dt, maxiter_eta=maxiter_eta,
                                          full_unroll_steps=full_unroll_steps,
                                          maxiter_outer=maxiter_outer)
                theta_hat = outcome.theta
                ad_eval, ad_eval_max_eta_grad, ad_eval_n_conv =
                    canonical_method(method) == "LAPLACE_IMPLICIT" ?
                    safe_laplace_population_evaluation(subjects, theta_hat, rep; dt=dt,
                                                       maxiter_eta=maxiter_eta) :
                    safe_ad_population_evaluation(subjects, theta_hat, rep; dt=dt,
                                                  maxiter_eta=maxiter_eta)
                if full_unroll_auto_retry && canonical_method(method) == "FULL_UNROLL"
                    current_linesearch = lowercase(get(ENV, method_env_key("FULL_UNROLL", "LINESEARCH"),
                                                       get(ENV, "WARFARIN_JULIA_LINESEARCH", "hagerzhang")))
                    mismatch = !isfinite(ad_eval) || !isfinite(outcome.objective) ||
                               abs(ad_eval - outcome.objective) > full_unroll_retry_delta
                    if current_linesearch != "backtracking" && mismatch
                        @printf("  FULL_UNROLL retry: ad_eval-objective mismatch %.6g, trying backtracking\n",
                                ad_eval - outcome.objective)
                        retry = with_env_value(method_env_key("FULL_UNROLL", "LINESEARCH"), "backtracking") do
                            optimize_method(method, subjects, rep, theta0, lo, hi;
                                            dt=dt, maxiter_eta=maxiter_eta,
                                            full_unroll_steps=full_unroll_steps,
                                            maxiter_outer=maxiter_outer)
                        end
                        retry_ad_eval, retry_ad_max_eta_grad, retry_ad_n_conv =
                            safe_ad_population_evaluation(subjects, retry.theta, rep; dt=dt,
                                                          maxiter_eta=maxiter_eta)
                        use_retry = isfinite(retry_ad_eval) && (!isfinite(ad_eval) || retry_ad_eval < ad_eval)
                        outcome = account_full_unroll_retry(outcome, retry; use_retry=use_retry)
                        theta_hat = outcome.theta
                        if use_retry
                            ad_eval = retry_ad_eval
                            ad_eval_max_eta_grad = retry_ad_max_eta_grad
                            ad_eval_n_conv = retry_ad_n_conv
                        end
                    end
                end
                stop_eval = ad_eval
                row = (
                    model="warfarin",
                    implementation=canonical_method(method) == "FD" ? "julia_finite_difference" : canonical_method(method) == "ALMQUIST_FORWARD" ? "julia_almquist_forward_sensitivity" : canonical_method(method) == "FULL_IMPLICIT_DIRECTIONAL_JVP" ? "julia_directional_jvp_contraction" : canonical_method(method) == "FULL_IMPLICIT_REVERSE_VJP" ? "julia_reverse_mode_vjp" : startswith(canonical_method(method), "HYBRID_") ? "julia_hybrid_reverse_vjp" : canonical_method(method) == "REVERSE_DIRECT_FORWARD_CONTRACTION" ? "julia_reverse_direct_forward_contraction" : canonical_method(method) == "LAPLACE_IMPLICIT" ? "laplace_companion_adapted_combined" : "julia_adjoint_contraction",
                    representation=string(rep),
                    method=outcome.method,
                    start_id=s - 1,
                    n_subjects=length(subjects),
                    maxiter_eta=maxiter_eta,
                    full_unroll_steps=full_unroll_steps,
                    outer_iterations=outcome.iterations,
                    success=outcome.converged && !outcome.failed && isfinite(ad_eval) &&
                            isfinite(outcome.max_eta_grad) && outcome.n_converged == length(subjects) &&
                            isfinite(ad_eval_max_eta_grad) && ad_eval_n_conv == length(subjects),
                    objective=outcome.objective,
                    objective_ad_eval=ad_eval,
                    objective_stop_eval=stop_eval,
                    ad_eval_max_eta_grad_norm=ad_eval_max_eta_grad,
                    ad_eval_n_eta_converged=ad_eval_n_conv,
                    wall_sec=outcome.wall,
                    cpu_sec=outcome.cpu,
                    method_eval_count=outcome.evals,
                    max_eta_grad_norm=outcome.max_eta_grad,
                    n_eta_converged=outcome.n_converged,
                    stage1_method=outcome.stage1_method,
                    stage1_outer=outcome.stage1_outer,
                    stage2_method=outcome.stage2_method,
                    stage2_outer=outcome.stage2_outer,
                    theta=theta_hat,
                )
                push!(rows, row)
                write_results(outpath, rows)
                @printf("  success=%s objective=%.6f ad_eval=%.6f wall=%.3f cpu=%.3f evals=%d\n",
                        string(row.success), row.objective, row.objective_ad_eval,
                        row.wall_sec, row.cpu_sec, row.method_eval_count)
            end
        end
    end
    println("\nSaved ", length(rows), " rows to ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
