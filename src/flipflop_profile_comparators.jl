# Matched comparison implementations.
# The Laplace functions below were adapted from the accepted companion paper's
# full-implicit implementation. They retain its exact-Hessian determinant,
# while using this study's combined residual-error likelihood and shared EBE solver.

function laplace_subject_fixed_eta(subj::SubjectData, theta, eta)
    hi = subject_nll(subj, theta, eta)
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    return 2.0 * (hi + 0.5 * logdet_cholesky(H))
end

function laplace_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    value = laplace_subject_fixed_eta(subj, theta, eta)
    dh_dtheta = ForwardDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dtheta = ForwardDiff.gradient(
        x -> logdet_cholesky(ForwardDiff.hessian(e -> subject_nll(subj, x, e), eta)), theta)
    dld_deta = ForwardDiff.gradient(
        e -> logdet_cholesky(ForwardDiff.hessian(ee -> subject_nll(subj, theta, ee), e)), eta)
    lambda = solve_mode_hessian(H, 0.5 .* Vector{Float64}(dld_deta))
    contraction = ForwardDiff.gradient(
        x -> dot(ForwardDiff.gradient(e -> subject_nll(subj, x, e), eta), lambda), theta)
    gradient = 2.0 .* (dh_dtheta .+ 0.5 .* dld_dtheta .- contraction)
    return Float64(value), Vector{Float64}(gradient)
end

function laplace_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                            eta_solver::Symbol=:newton,
                            eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = laplace_subject_value_grad(subjects[i], theta, etas[i])
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end

function safe_laplace_population_evaluation(subjects::Vector{SubjectData}, theta::Vector{Float64};
                                            maxiter_eta::Int=20, eta_solver::Symbol=:newton)
    try
        etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                    backend=:forward, solver=eta_solver)
        values = zeros(length(subjects))
        @threads for i in eachindex(subjects)
            values[i] = Float64(laplace_subject_fixed_eta(subjects[i], theta, etas[i]))
        end
        return sum(values), max_eta_grad, n_conv
    catch err
        @warn "Laplace endpoint re-evaluation failed; recording NaN" exception=typeof(err)
        return NaN, NaN, 0
    end
end

# Almquist et al.'s forward organization of the converged-mode FOCEI derivative.
# B is the q-by-p mixed score derivative and S=-H^{-1}B collects the q-by-p
# EBE sensitivities. This uses ForwardDiff to obtain the model derivatives,
# but deliberately materializes all parameter directions rather than using
# the adjoint scalar contraction.
function almquist_forward_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H = ForwardDiff.hessian(e -> subject_nll(subj, theta, e), eta)
    value = focei_subject_fixed_eta(subj, theta, eta)
    dh_dtheta = ForwardDiff.gradient(x -> subject_nll(subj, x, eta), theta)
    dld_dtheta = ForwardDiff.gradient(x -> logdet_cholesky(focei_curvature(subj, x, eta)), theta)
    dld_deta = ForwardDiff.gradient(e -> logdet_cholesky(focei_curvature(subj, theta, e)), eta)
    B = ForwardDiff.jacobian(
        x -> ForwardDiff.gradient(e -> subject_nll(subj, x, e), eta), theta)
    Hs = Matrix{Float64}(0.5 .* (H .+ transpose(H)))
    F = cholesky(Symmetric(Hs); check=true)
    S = -(F \ Matrix{Float64}(B))
    gradient = 2.0 .* (dh_dtheta .+ 0.5 .* dld_dtheta .+
                       transpose(S) * (0.5 .* Vector{Float64}(dld_deta)))
    return Float64(value), Vector{Float64}(gradient)
end

function almquist_forward_value_grad(subjects::Vector{SubjectData}, theta::Vector{Float64}; maxiter_eta::Int=20,
                                     eta_solver::Symbol=:newton,
                                     eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas, max_eta_grad, n_conv = solve_all_etas(subjects, theta; maxiter=maxiter_eta,
                                                backend=:forward, solver=eta_solver,
                                                eta_cache=eta_cache)
    values = zeros(length(subjects))
    gradients = [zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects)
        values[i], gradients[i] = almquist_forward_subject_value_grad(subjects[i], theta, etas[i])
    end
    total = sum(values)
    total_gradient = vec(sum(reduce(hcat, gradients), dims=2))
    maybe_update_eta_cache!(eta_cache, etas, total, total_gradient)
    return total, total_gradient, max_eta_grad, n_conv
end
# Compute B' * lambda with one eta-directional JVP rather than constructing
# the full mixed score derivative.  This is exact for fixed lambda and uses
# only one eta tangent direction inside the outer population derivative.
function directional_score_contraction(subj::SubjectData, theta, eta::Vector{Float64}, lambda::Vector{Float64})
    return ForwardDiff.derivative(t -> subject_nll(subj, theta, eta .+ t .* lambda), 0.0)
end
function full_implicit_directional_jvp_subject_value_grad(subj::SubjectData, theta::Vector{Float64}, eta::Vector{Float64})
    H=ForwardDiff.hessian(e->subject_nll(subj,theta,e),eta)
    value=focei_subject_fixed_eta(subj,theta,eta)
    dh=ForwardDiff.gradient(x->subject_nll(subj,x,eta),theta)
    dldx=ForwardDiff.gradient(x->logdet_cholesky(focei_curvature(subj,x,eta)),theta)
    dlde=ForwardDiff.gradient(e->logdet_cholesky(focei_curvature(subj,theta,e)),eta)
    lambda=solve_mode_hessian(H,0.5 .* Vector{Float64}(dlde))
    contraction=ForwardDiff.gradient(x->directional_score_contraction(subj,x,eta,lambda),theta)
    return Float64(value),Vector{Float64}(2.0 .* (dh .+ .5 .* dldx .- contraction))
end
function full_implicit_directional_jvp_value_grad(subjects::Vector{SubjectData},theta::Vector{Float64};maxiter_eta::Int=20,eta_solver::Symbol=:newton,eta_cache::Union{Nothing,EtaWarmCache}=nothing)
    etas,max_eta_grad,n_conv=solve_all_etas(subjects,theta;maxiter=maxiter_eta,backend=:forward,solver=eta_solver,eta_cache=eta_cache)
    values=zeros(length(subjects)); gradients=[zeros(length(theta)) for _ in subjects]
    @threads for i in eachindex(subjects); values[i],gradients[i]=full_implicit_directional_jvp_subject_value_grad(subjects[i],theta,etas[i]); end
    total=sum(values); total_gradient=vec(sum(reduce(hcat,gradients),dims=2)); maybe_update_eta_cache!(eta_cache,etas,total,total_gradient)
    return total,total_gradient,max_eta_grad,n_conv
end