# Matched comparison implementations.
# The Laplace functions below were adapted from the accepted companion paper's
# full-implicit implementation. They retain its exact-Hessian determinant,
# while using this study's combined PK and PD residual-error likelihood and
# shared EBE solver.

function laplace_subject_fixed_eta(subj::SubjectData, x, eta, representation::Symbol; dt::Float64=0.25)
    hi = h_i(subj, x, eta, representation; dt=dt)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky_ad(H))
end

function laplace_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                    representation::Symbol; dt::Float64=0.25)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    value = laplace_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    dh_dx = ForwardDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(
        xx -> logdet_cholesky_ad(ForwardDiff.hessian(e -> h_i(subj, xx, e, representation; dt=dt), eta)), x)
    dld_deta = ForwardDiff.gradient(
        ee -> logdet_cholesky_ad(ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), ee)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    contraction = ForwardDiff.gradient(
        xx -> dot(ForwardDiff.gradient(e -> h_i(subj, xx, e, representation; dt=dt), eta), lambda), x)
    gradient = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .- contraction)
    return Float64(value), Vector{Float64}(gradient)
end

function laplace_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                            dt::Float64=0.25, maxiter_eta::Int=30,
                            eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = laplace_subject_value_grad(subjects[i], x, etas[i], representation; dt=dt)
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end

function safe_laplace_population_evaluation(subjects::Vector{SubjectData}, x::Vector{Float64},
                                            representation::Symbol; dt::Float64=0.25,
                                            maxiter_eta::Int=30)
    try
        etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                    maxiter=maxiter_eta, backend=:forward)
        values = zeros(length(subjects))
        @threads for i in eachindex(subjects)
            values[i] = Float64(laplace_subject_fixed_eta(subjects[i], x, etas[i], representation; dt=dt))
        end
        return sum(values), max_eta_grad, n_conv
    catch err
        @warn "Laplace endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN, NaN, 0
    end
end

# Almquist et al.'s forward organization of the converged-mode FOCEI derivative.
# B is the q-by-p mixed score derivative and S=-H^{-1}B collects all EBE
# sensitivities. ForwardDiff supplies model derivatives, while this method
# intentionally materializes each population-parameter direction.
function almquist_forward_subject_value_grad(subj::SubjectData, x::Vector{Float64}, eta::Vector{Float64},
                                             representation::Symbol; dt::Float64=0.25)
    H = ForwardDiff.hessian(e -> h_i(subj, x, e, representation; dt=dt), eta)
    value = focei_subject_fixed_eta(subj, x, eta, representation; dt=dt)
    dh_dx = ForwardDiff.gradient(xx -> h_i(subj, xx, eta, representation; dt=dt), x)
    dld_dx = ForwardDiff.gradient(
        xx -> logdet_cholesky_ad(focei_curvature(subj, xx, eta, representation; dt=dt)), x)
    dld_deta = ForwardDiff.gradient(
        ee -> logdet_cholesky_ad(focei_curvature(subj, x, ee, representation; dt=dt)), eta)
    B = ForwardDiff.jacobian(
        xx -> ForwardDiff.gradient(e -> h_i(subj, xx, e, representation; dt=dt), eta), x)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    F = cholesky(Symmetric(Hs); check=true)
    S = -(F \ Matrix{Float64}(B))
    gradient = 2.0 .* (dh_dx .+ 0.5 .* dld_dx .+
                       transpose(S) * (0.5 .* Vector{Float64}(dld_deta)))
    return Float64(value), Vector{Float64}(gradient)
end

function almquist_forward_value_grad(subjects::Vector{SubjectData}, x::Vector{Float64}, representation::Symbol;
                                     dt::Float64=0.25, maxiter_eta::Int=30,
                                     eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, x, representation; dt=dt,
                                                maxiter=maxiter_eta, backend=:forward,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = almquist_forward_subject_value_grad(subjects[i], x, etas[i], representation; dt=dt)
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, subjects, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end
# Compute B' * lambda with one eta-directional JVP rather than constructing
# the full mixed score derivative. This is exact for fixed lambda.
function directional_score_contraction(subj::SubjectData, x, eta::Vector{Float64}, lambda::Vector{Float64}, representation::Symbol;dt::Float64=0.25)
    return ForwardDiff.derivative(t -> h_i(subj,x,eta .+ t .* lambda,representation;dt=dt),0.0)
end
function full_implicit_directional_jvp_subject_value_grad(subj::SubjectData,x::Vector{Float64},eta::Vector{Float64},representation::Symbol;dt::Float64=0.25)
    H=ForwardDiff.hessian(e->h_i(subj,x,e,representation;dt=dt),eta)
    value=focei_subject_fixed_eta(subj,x,eta,representation;dt=dt)
    dh=ForwardDiff.gradient(xx->h_i(subj,xx,eta,representation;dt=dt),x)
    dldx=ForwardDiff.gradient(xx->logdet_cholesky_ad(focei_curvature(subj,xx,eta,representation;dt=dt)),x)
    dlde=ForwardDiff.gradient(e->logdet_cholesky_ad(focei_curvature(subj,x,e,representation;dt=dt)),eta)
    lambda=solve_mode_hessian(H,0.5 .* Vector{Float64}(dlde))
    contraction=ForwardDiff.gradient(xx->directional_score_contraction(subj,xx,eta,lambda,representation;dt=dt),x)
    return Float64(value),Vector{Float64}(2.0 .* (dh .+ .5 .* dldx .- contraction))
end
function full_implicit_directional_jvp_value_grad(subjects::Vector{SubjectData},x::Vector{Float64},representation::Symbol;dt::Float64=0.25,maxiter_eta::Int=30,eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas,max_eta_grad,n_conv=solve_all_etas(subjects,x,representation;dt=dt,maxiter=maxiter_eta,backend=:forward,eta_cache=eta_cache)
    values=zeros(length(subjects)); gradients=[zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects); values[i],gradients[i]=full_implicit_directional_jvp_subject_value_grad(subjects[i],x,etas[i],representation;dt=dt); end
    total=sum(values); total_gradient=vec(sum(reduce(hcat,gradients),dims=2)); maybe_update_eta_cache!(eta_cache,subjects,etas,total,total_gradient)
    return total,total_gradient,max_eta_grad,n_conv
end