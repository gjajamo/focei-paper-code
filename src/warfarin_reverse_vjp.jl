# Reverse-mode VJP implementation for the combined-error warfarin FOCEI model.
# The eta prediction Jacobians are propagated analytically through the RK4
# effect-compartment solver. All population partial derivatives are ReverseDiff
# pullbacks; this path contains no ForwardDiff calls.

function _reverse_warfarin_pk_value_eta_jac(t, dose, ka, cl, v)
    k = cl / v
    d = ka - k
    z = d * t
    ek = exp(-k * t)
    ea = exp(-ka * t)
    if abs(primal_float(z)) < 1.0e-6
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
    mu = (dose / v) * ka * q
    dmu_dka = (dose / v) * (q + ka * dq_dka)
    dmu_dk = (dose / v) * ka * dq_dk
    dmu_dcl = dmu_dk / v
    dmu_dv = -mu / v - dmu_dk * k / v
    T = promote_type(typeof(mu), typeof(ka), typeof(cl), typeof(v))
    J = zeros(T, ETA_DIM)
    J[1] = dmu_dka * ka
    J[2] = dmu_dcl * cl
    J[3] = dmu_dv * v
    return mu, J
end

function _reverse_ce_eta_jac(times::Vector{Float64}, dose, ka, cl, v, ke0; dt_max::Float64=0.25)
    T = promote_type(typeof(dose), typeof(ka), typeof(cl), typeof(v), typeof(ke0))
    ce = zero(T)
    sensitivity = zeros(T, ETA_DIM)
    current_t = 0.0
    outputs = Vector{T}(undef, length(times))
    output_sensitivity = Matrix{T}(undef, length(times), ETA_DIM)
    function rhs(t, ce_value, s_value)
        conc, conc_eta = _reverse_warfarin_pk_value_eta_jac(t, dose, ka, cl, v)
        dce = ke0 * (conc - ce_value)
        ds = zeros(T, ETA_DIM)
        for j in 1:ETA_DIM
            ds[j] = ke0 * (conc_eta[j] - s_value[j])
        end
        ds[6] += ke0 * (conc - ce_value)
        return dce, ds
    end
    for (row, target) in enumerate(times)
        interval = max(target - current_t, 0.0)
        if interval > 0.0
            n_steps = max(1, ceil(Int, interval / dt_max))
            h = interval / n_steps
            for _ in 1:n_steps
                half_h = h / 2
                k1, s1 = rhs(current_t, ce, sensitivity)
                k2, s2 = rhs(current_t + half_h, ce + half_h * k1, sensitivity .+ half_h .* s1)
                k3, s3 = rhs(current_t + half_h, ce + half_h * k2, sensitivity .+ half_h .* s2)
                k4, s4 = rhs(current_t + h, ce + h * k3, sensitivity .+ h .* s3)
                ce += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                sensitivity .+= (h / 6) .* (s1 .+ 2 .* s2 .+ 2 .* s3 .+ s4)
                current_t += h
            end
        end
        outputs[row] = ce
        for j in 1:ETA_DIM
            output_sensitivity[row, j] = sensitivity[j]
        end
    end
    return outputs, output_sensitivity
end

function reverse_warfarin_predictions_eta_jac(subj::SubjectData, x, eta, representation::Symbol;
                                               dt::Float64=0.25)
    representation == :ode || error("reverse VJP comparator currently supports the ODE representation")
    ka = exp(x[1] + eta[1]); cl = exp(x[2] + eta[2]); v = exp(x[3] + eta[3])
    e0 = exp(x[4] + eta[4]); c50 = exp(x[5] + eta[5]); ke0 = exp(x[6] + eta[6])
    emax = sigmoid(x[7])
    dose = typeof(ka + cl + v + e0 + c50 + ke0)(subj.dose_mg)
    T = promote_type(typeof(ka), typeof(cl), typeof(v), typeof(e0), typeof(c50), typeof(ke0))
    pk = Vector{T}(undef, length(subj.pk_times))
    Jpk = Matrix{T}(undef, length(subj.pk_times), ETA_DIM)
    for (row, t) in enumerate(subj.pk_times)
        pk[row], jac_row = _reverse_warfarin_pk_value_eta_jac(t, dose, ka, cl, v)
        for col in 1:ETA_DIM
            Jpk[row, col] = jac_row[col]
        end
    end
    ce, Jce = _reverse_ce_eta_jac(subj.pd_times, dose, ka, cl, v, ke0; dt_max=dt)
    pd = Vector{T}(undef, length(ce))
    Jpd = Matrix{T}(undef, length(ce), ETA_DIM)
    for row in eachindex(ce)
        c = ce[row]
        denom = c50 + c + 1.0e-12
        frac = c / denom
        base = one(c) - emax * frac
        pd[row] = e0 * base
        for j in 1:ETA_DIM
            dc50 = j == 5 ? c50 : zero(T)
            de0 = j == 4 ? e0 : zero(T)
            dfrac = (Jce[row, j] * c50 - c * dc50) / (denom * denom)
            Jpd[row, j] = de0 * base - e0 * emax * dfrac
        end
    end
    return pk, Jpk, pd, Jpd
end

function reverse_warfarin_score(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    pk, Jpk, pd, Jpd = reverse_warfarin_predictions_eta_jac(subj, x, eta, representation; dt=dt)
    sigma_add_pk = exp(x[8]); sigma_add_pd = exp(x[9])
    combined = warfarin_combined_error()
    sigma_prop_pk = combined ? exp(x[10]) : zero(sigma_add_pk)
    sigma_prop_pd = combined ? exp(x[11]) : zero(sigma_add_pd)
    omega_start = combined ? 12 : 10
    omega = exp.(x[omega_start:(omega_start + ETA_DIM - 1)])
    T = promote_type(eltype(pk), eltype(pd), eltype(eta))
    score = zeros(T, ETA_DIM)
    for row in eachindex(pk)
        mu = pk[row]; variance = warfarin_variance(mu, sigma_add_pk, sigma_prop_pk)
        residual = subj.pk_obs[row] - mu
        dloss_dmu = -residual / variance + 0.5 * (inv(variance) - residual^2 / (variance * variance)) *
                      (2 * sigma_prop_pk^2 * mu)
        for j in 1:ETA_DIM; score[j] += dloss_dmu * Jpk[row, j]; end
    end
    for row in eachindex(pd)
        mu = pd[row]; variance = warfarin_variance(mu, sigma_add_pd, sigma_prop_pd)
        residual = subj.pd_obs[row] - mu
        dloss_dmu = -residual / variance + 0.5 * (inv(variance) - residual^2 / (variance * variance)) *
                      (2 * sigma_prop_pd^2 * mu)
        for j in 1:ETA_DIM; score[j] += dloss_dmu * Jpd[row, j]; end
    end
    for j in 1:ETA_DIM; score[j] += eta[j] / (omega[j] * omega[j]); end
    return score
end

function reverse_warfarin_focei_curvature(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    pk, Jpk, pd, Jpd = reverse_warfarin_predictions_eta_jac(subj, x, eta, representation; dt=dt)
    sigma_add_pk = exp(x[8]); sigma_add_pd = exp(x[9])
    combined = warfarin_combined_error()
    sigma_prop_pk = combined ? exp(x[10]) : zero(sigma_add_pk)
    sigma_prop_pd = combined ? exp(x[11]) : zero(sigma_add_pd)
    omega_start = combined ? 12 : 10
    omega = exp.(x[omega_start:(omega_start + ETA_DIM - 1)])
    T = promote_type(eltype(pk), eltype(pd), eltype(Jpk), eltype(Jpd))
    G = zeros(T, ETA_DIM, ETA_DIM)
    for (predictions, jacobian, sigma_add, sigma_prop) in
        ((pk, Jpk, sigma_add_pk, sigma_prop_pk), (pd, Jpd, sigma_add_pd, sigma_prop_pd))
        for row in eachindex(predictions)
            mu = predictions[row]; variance = warfarin_variance(mu, sigma_add, sigma_prop)
            dvariance = 2 * sigma_prop^2 * mu
            for a in 1:ETA_DIM, b in 1:ETA_DIM
                ja = jacobian[row, a]; jb = jacobian[row, b]
                G[a, b] += ja * jb / variance +
                           0.5 * (dvariance * ja) * (dvariance * jb) / (variance * variance)
            end
        end
    end
    for j in 1:ETA_DIM; G[j, j] += one(T) / (omega[j] * omega[j]); end
    return G
end

reverse_warfarin_logdet(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25) =
    logdet_cholesky_ad(reverse_warfarin_focei_curvature(subj, x, eta, representation; dt=dt))

function full_implicit_reverse_vjp_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                                       representation::Symbol; dt::Float64=0.25)
    value = 2.0 * (h_i(subj, x, eta, representation; dt=dt) + 0.5 * reverse_warfarin_logdet(subj, x, eta, representation; dt=dt))
    H = ReverseDiff.jacobian(e -> reverse_warfarin_score(subj, x, e, representation; dt=dt), eta)
    dh_dx = ReverseDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ReverseDiff.gradient(xx -> reverse_warfarin_logdet(subj, xx, eta, representation; dt=dt), x)
    dld_deta = ReverseDiff.gradient(e -> reverse_warfarin_logdet(subj, x, e, representation; dt=dt), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    score_vjp = ReverseDiff.gradient(xx -> dot(lambda, reverse_warfarin_score(subj, xx, eta, representation; dt=dt)), x)
    gradient = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- score_vjp)
    return Float64(value), Vector{Float64}(gradient)
end

function full_implicit_reverse_vjp_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                              dt::Float64=0.25, maxiter_eta::Int=30,
                                              eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = full_implicit_reverse_vjp_subject_value_grad(subjects[i], x, etas[i], representation; dt=dt)
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end
# Hybrid implicit gradients used to isolate the cost of the adjoint VJP.
# The FOCEI determinant remains on the established ForwardDiff curvature path;
# only the contraction lambda' * score_theta is evaluated as a reverse VJP.
function _hybrid_implicit_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                             representation::Symbol; dt::Float64=0.25,
                                             direct_backend::Symbol=:forward,
                                             hessian_backend::Symbol=:forward)
    value = focei_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    H = hessian_backend == :forward ?
        ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta) :
        ReverseDiff.jacobian(e -> reverse_warfarin_score(subj, x, e, representation; dt=dt), eta)
    dh_dx = direct_backend == :forward ?
        ForwardDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x) :
        ReverseDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(
        xx -> logdet_cholesky_ad(focei_curvature(subj, xx, eta, representation; dt=dt)), x)
    dld_deta = ForwardDiff.gradient(
        e -> logdet_cholesky_ad(focei_curvature(subj, x, e, representation; dt=dt)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    score_vjp = ReverseDiff.gradient(
        xx -> dot(lambda, reverse_warfarin_score(subj, xx, eta, representation; dt=dt)), x)
    gradient = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- score_vjp)
    return Float64(value), Vector{Float64}(gradient)
end

function _hybrid_implicit_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                     dt::Float64=0.25, direct_backend::Symbol=:forward,
                                     hessian_backend::Symbol=:forward, maxiter_eta::Int=30,
                                     eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = _hybrid_implicit_subject_value_grad(
            subjects[i], x, etas[i], representation; dt=dt,
            direct_backend=direct_backend, hessian_backend=hessian_backend)
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end

full_implicit_hybrid_ff_reverse_vjp_value_grad(subjects, x, representation; kwargs...) =
    _hybrid_implicit_value_grad(subjects, x, representation; direct_backend=:forward, hessian_backend=:forward, kwargs...)
full_implicit_hybrid_rf_reverse_vjp_value_grad(subjects, x, representation; kwargs...) =
    _hybrid_implicit_value_grad(subjects, x, representation; direct_backend=:reverse, hessian_backend=:forward, kwargs...)
full_implicit_hybrid_rrh_reverse_vjp_value_grad(subjects, x, representation; kwargs...) =
    _hybrid_implicit_value_grad(subjects, x, representation; direct_backend=:reverse, hessian_backend=:reverse, kwargs...)
# Reverse-forward hybrid: reverse only the scalar direct likelihood partial;
# retain the established forward G and implicit-contraction derivatives.
function full_implicit_reverse_direct_forward_contraction_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64}, representation::Symbol; dt::Float64=0.25)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    value = focei_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    dh_dx = ReverseDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(xx -> logdet_cholesky_ad(focei_curvature(subj, xx, eta, representation; dt=dt)), x)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky_ad(focei_curvature(subj, x, e, representation; dt=dt)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    term_c = ForwardDiff.gradient(xx -> dot(ForwardDiff.gradient(e -> h_i(subj, xx, e, representation; dt=dt), eta), lambda), x)
    gradient = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- term_c)
    return Float64(value), Vector{Float64}(gradient)
end
function full_implicit_reverse_direct_forward_contraction_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol; dt::Float64=0.25, maxiter_eta::Int=30, eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas,max_eta_grad,n_conv=solve_all_etas(subjects,x,representation;dt=dt,maxiter=maxiter_eta,backend=:forward,eta_cache=eta_cache)
    values=zeros(length(subjects)); gradients=[zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i],gradients[i]=full_implicit_reverse_direct_forward_contraction_subject_value_grad(subjects[i],x,etas[i],representation;dt=dt)
    end
    total=sum(values); total_gradient=vec(sum(reduce(hcat,gradients),dims=2)); maybe_update_eta_cache!(eta_cache,subjects,etas,total,total_gradient)
    return total,total_gradient,max_eta_grad,n_conv
end