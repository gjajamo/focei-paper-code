# A reverse-mode VJP implementation of the FOCEI implicit derivative.
#
# Fixed-EBE partial derivatives and the score VJP below use ReverseDiff. The
# eta prediction derivatives needed for G are written analytically, avoiding
# nested ForwardDiff calls and the q-by-p forward-sensitivity matrix.

function _reverse_pk_value_eta_jac(t, ka, cl, v)
    k = cl / v
    d = ka - k
    z = d * t
    ek = exp(-k * t)
    ea = exp(-ka * t)
    if abs(base_float(z)) < 1.0e-6
        f = one(z) - z / 2 + z^2 / 6 - z^3 / 24 + z^4 / 120
        fp = -one(z) / 2 + z / 3 - z^2 / 8 + z^3 / 30
        q = ek * t * f
        dq_dka = ek * t * fp * t
        dq_dk = -ek * t^2 * (f + fp)
    else
        q = (ek - ea) / d
        dq_dka = (t * ea * d - (ek - ea)) / (d * d)
        dq_dk = (-t * ek * d + (ek - ea)) / (d * d)
    end
    mu = (DOSE / v) * ka * q
    dmu_dka = (DOSE / v) * (q + ka * dq_dka)
    dmu_dk = (DOSE / v) * ka * dq_dk
    dmu_dv = -mu / v - dmu_dk * k / v
    T = promote_type(typeof(mu), typeof(ka), typeof(v))
    J = Vector{T}(undef, ETA_DIM)
    J[1] = dmu_dka * ka
    J[2] = dmu_dv * v
    return mu, J
end

function reverse_predictions_eta_jac(subj::SubjectData, theta, eta)
    pars = unpack_theta(theta)
    ka = exp(pars.logka + eta[1])
    cl = exp(pars.logcl)
    v = exp(pars.logv + eta[2])
    T = promote_type(typeof(ka), typeof(cl), typeof(v))
    predictions = Vector{T}(undef, length(TIMES))
    J = Matrix{T}(undef, length(TIMES), ETA_DIM)
    for (row, t) in enumerate(TIMES)
        predictions[row], jac_row = _reverse_pk_value_eta_jac(t, ka, cl, v)
        for col in 1:ETA_DIM
            J[row, col] = jac_row[col]
        end
    end
    return predictions, J
end

function reverse_focei_curvature(subj::SubjectData, theta, eta)
    predictions, J = reverse_predictions_eta_jac(subj, theta, eta)
    pars = unpack_theta(theta)
    T = promote_type(eltype(predictions), eltype(J))
    G = zeros(T, ETA_DIM, ETA_DIM)
    for row in eachindex(predictions)
        mu = predictions[row]
        variance = residual_variance(mu, pars.sigma_add, pars.sigma_prop)
        dvariance = 2 * pars.sigma_prop^2 * mu
        for a in 1:ETA_DIM, b in 1:ETA_DIM
            ja = J[row, a]
            jb = J[row, b]
            G[a, b] += ja * jb / variance +
                       0.5 * (dvariance * ja) * (dvariance * jb) / (variance * variance)
        end
    end
    G[1, 1] += one(T) / (pars.omega_ka * pars.omega_ka)
    G[2, 2] += one(T) / (pars.omega_v * pars.omega_v)
    return G
end

reverse_focei_logdet(subj::SubjectData, theta, eta) =
    logdet_cholesky(reverse_focei_curvature(subj, theta, eta))

# Analytic conditional score. ReverseDiff differentiates this vector-valued
# function to obtain H and the VJP lambda' * d(score)/d(theta).
function reverse_subject_score(subj::SubjectData, theta, eta)
    predictions, J = reverse_predictions_eta_jac(subj, theta, eta)
    pars = unpack_theta(theta)
    T = promote_type(eltype(predictions), eltype(J), eltype(eta))
    score = zeros(T, ETA_DIM)
    for row in eachindex(predictions)
        mu = predictions[row]
        variance = residual_variance(mu, pars.sigma_add, pars.sigma_prop)
        residual = subj.y[row] - mu
        dvariance_dmu = 2 * pars.sigma_prop^2 * mu
        dloss_dmu = -residual / variance +
                      0.5 * (inv(variance) - residual^2 / (variance * variance)) * dvariance_dmu
        for a in 1:ETA_DIM
            score[a] += dloss_dmu * J[row, a]
        end
    end
    score[1] += eta[1] / (pars.omega_ka * pars.omega_ka)
    score[2] += eta[2] / (pars.omega_v * pars.omega_v)
    return score
end

function full_implicit_reverse_vjp_subject_value_grad(subj::SubjectData,
                                                       theta::Vector{Float64},
                                                       eta::Vector{Float64})
    value = 2.0 * (subject_nll(subj, theta, eta) + 0.5 * reverse_focei_logdet(subj, theta, eta))
    H = ReverseDiff.jacobian(e -> reverse_subject_score(subj, theta, e), eta)
    dh_dtheta = ReverseDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dtheta = ReverseDiff.gradient(x -> reverse_focei_logdet(subj, x, eta), theta)
    dld_deta = ReverseDiff.gradient(e -> reverse_focei_logdet(subj, theta, e), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    score_vjp = ReverseDiff.gradient(x -> dot(lambda, reverse_subject_score(subj, x, eta)), theta)
    gradient = 2.0 .* (dh_dtheta .+ 0.5 .* dld_dtheta .- score_vjp)
    return Float64(value), Vector{Float64}(gradient)
end

function full_implicit_reverse_vjp_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64};
                                              maxiter_eta::Int=20,
                                              eta_solver::Symbol=:newton,
                                              eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = full_implicit_reverse_vjp_subject_value_grad(subjects[i], theta, etas[i])
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end
# Hybrid implicit gradients used to isolate the cost of the adjoint VJP.
# G and its determinant partials retain the existing ForwardDiff construction,
# while the score contraction is a reverse-mode VJP of the analytic score.
function _hybrid_implicit_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64};
                                             direct_backend::Symbol=:forward,
                                             hessian_backend::Symbol=:forward)
    value = focei_subject_fixed_eta(subj, theta, eta)
    H = hessian_backend == :forward ?
        ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta) :
        ReverseDiff.jacobian(e -> reverse_subject_score(subj, theta, e), eta)
    dh_dtheta = direct_backend == :forward ?
        ForwardDiff.gradient(x -> subject_nll(subj, x, eta), theta) :
        ReverseDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dtheta = ForwardDiff.gradient(x -> logdet_cholesky(focei_curvature(subj, x, eta)), theta)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky(focei_curvature(subj, theta, e)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    score_vjp = ReverseDiff.gradient(x -> dot(lambda, reverse_subject_score(subj, x, eta)), theta)
    gradient = 2.0 .* (dh_dtheta .+ 0.5 .* dld_dtheta .- score_vjp)
    return Float64(value), Vector{Float64}(gradient)
end

function _hybrid_implicit_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64};
                                     direct_backend::Symbol=:forward,
                                     hessian_backend::Symbol=:forward,
                                     maxiter_eta::Int=20, eta_solver::Symbol=:newton,
                                     eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = _hybrid_implicit_subject_value_grad(
            subjects[i], theta, etas[i]; direct_backend=direct_backend,
            hessian_backend=hessian_backend)
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end

full_implicit_hybrid_ff_reverse_vjp_value_grad(subjects, theta; kwargs...) =
    _hybrid_implicit_value_grad(subjects, theta; direct_backend=:forward, hessian_backend=:forward, kwargs...)
full_implicit_hybrid_rf_reverse_vjp_value_grad(subjects, theta; kwargs...) =
    _hybrid_implicit_value_grad(subjects, theta; direct_backend=:reverse, hessian_backend=:forward, kwargs...)
full_implicit_hybrid_rrh_reverse_vjp_value_grad(subjects, theta; kwargs...) =
    _hybrid_implicit_value_grad(subjects, theta; direct_backend=:reverse, hessian_backend=:reverse, kwargs...)
# Reverse-forward hybrid: reverse only the scalar direct likelihood partial;
# retain the established forward G and implicit-contraction derivatives.
function full_implicit_reverse_direct_forward_contraction_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    value = focei_subject_fixed_eta(subj, theta, eta)
    dh_dtheta = ReverseDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dtheta = ForwardDiff.gradient(x -> logdet_cholesky(focei_curvature(subj, x, eta)), theta)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky(focei_curvature(subj, theta, e)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(x -> dot(ForwardDiff.gradient(e -> subject_nll(subj, x, e), eta), lambda), theta)
    gradient = 2.0 .* (dh_dtheta .+ 0.5 .* dld_dtheta .- term_c)
    return Float64(value), Vector{Float64}(gradient)
end
function full_implicit_reverse_direct_forward_contraction_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20, eta_solver::Symbol=:newton, eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas,max_eta_grad,n_conv=solve_all_etas(subjects,theta;maxiter=maxiter_eta,backend=:forward,solver=eta_solver,eta_cache=eta_cache)
    values=zeros(length(subjects)); gradients=[zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i],gradients[i]=full_implicit_reverse_direct_forward_contraction_subject_value_grad(subjects[i],theta,etas[i])
    end
    total=sum(values); total_gradient=vec(sum(reduce(hcat,gradients),dims=2)); maybe_update_eta_cache!(eta_cache,etas,total,total_gradient)
    return total,total_gradient,max_eta_grad,n_conv
end