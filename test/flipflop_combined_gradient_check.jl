using LinearAlgebra
using ForwardDiff

ENV["FLIPFLOP_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["FLIPFLOP_JULIA_USE_ETA_CACHE"] = "false"
ENV["FLIPFLOP_JULIA_FULL_UNROLL_USE_ETA_CACHE"] = "false"
ENV["FLIPFLOP_JULIA_LOGDET_MODE"] = "raw"
ENV["FLIPFLOP_JULIA_OUTER_AD_MODE"] = "forward"
ENV["FLIPFLOP_JULIA_FD_REQUIRE_PERTURBED_ETA_CONVERGENCE"] = "true"

include(joinpath(@__DIR__, "..", "src", "flipflop_multistart_methods.jl"))

RESIDUAL_MODEL == :combined || error("combined residual model was not activated")
theta = log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.03, 0.15])
subj = simulate_subjects(theta; n_subj=1, seed=321)[1]
direction = normalize([0.31, -0.27, 0.19, -0.23, 0.29, -0.17, 0.21])
hs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

function solve_checked(theta; eta0=zeros(ETA_DIM), score_tol=5e-7)
    eta, _, _, _ = eta_mode_newton(subj, Vector{Float64}(theta); maxiter=300,
                                    tol=1e-10, backend=:forward,
                                    eta0=Vector{Float64}(eta0))
    score = ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta)
    norm(score) <= score_tol || error("EBE score norm $(norm(score)) exceeds $score_tol")
    return eta
end

function directional_ladder(f, x, v)
    [(h=h, derivative=(f(x .+ h .* v) - f(x .- h .* v)) / (2h)) for h in hs]
end

function report_check(label, ad, rows; tol)
    errs = [abs(ad - r.derivative) / max(1.0, abs(ad), abs(r.derivative)) for r in rows]
    k = argmin(errs)
    println(label, ",ad=", ad, ",best_fd=", rows[k].derivative,
            ",best_h=", rows[k].h, ",relerr=", errs[k])
    minimum(errs) <= tol || error("$label relative error $(minimum(errs)) exceeds $tol")
end

eta = solve_checked(theta)
score_ad = ForwardDiff.gradient(e -> subject_nll(subj, theta, e), eta)
score_fd = zeros(ETA_DIM)
for j in 1:ETA_DIM
    h = 1e-6 * max(1.0, abs(eta[j]))
    ep = copy(eta)
    em = copy(eta)
    ep[j] += h
    em[j] -= h
    score_fd[j] = (subject_nll(subj, theta, ep) - subject_nll(subj, theta, em)) / (2h)
end
score_err = norm(score_ad - score_fd) / max(1.0, norm(score_ad), norm(score_fd))
println("COMBINED_SCORE,relerr=", score_err)
score_err <= 2e-7 || error("combined conditional score mismatch $score_err")

fixed_value = x -> focei_subject_fixed_eta(subj, x, eta)
g_stop = ForwardDiff.gradient(fixed_value, theta)
report_check("COMBINED_STOP_FIXED_ETA", dot(g_stop, direction),
             directional_ladder(fixed_value, theta, direction); tol=5e-6)

_, g_implicit = full_implicit_subject_value_grad(subj, theta, eta)
profile_value = x -> begin
    etax = solve_checked(x; eta0=eta)
    focei_subject_fixed_eta(subj, x, etax)
end
report_check("COMBINED_FULL_IMPLICIT_PROFILE", dot(g_implicit, direction),
             directional_ladder(profile_value, theta, direction); tol=5e-5)

eta0 = zeros(ETA_DIM)
unroll_value = x -> full_unroll_subject_value(subj, x, eta0; steps=30)
g_unroll = ForwardDiff.gradient(unroll_value, theta)
report_check("COMBINED_FULL_UNROLL_PATHWISE", dot(g_unroll, direction),
             directional_ladder(unroll_value, theta, direction); tol=3e-3)

G_ad = focei_curvature(subj, theta, eta)
G_fd = focei_curvature_fd(subj, theta, eta; h=1e-5)
gerr = norm(G_ad - G_fd) / max(1.0, norm(G_ad))
println("COMBINED_FOCEI_CURVATURE,relerr=", gerr)
gerr <= 5e-7 || error("combined FOCEI curvature mismatch $gerr")

predictions = focei_predictions(subj, theta, eta)
prediction_jacobian = ForwardDiff.jacobian(e -> focei_predictions(subj, theta, e), eta)
G_without_covariance_sensitivity =
    foce_curvature_from_components(theta, predictions, prediction_jacobian)
interaction = Symmetric(0.5 .* ((G_ad - G_without_covariance_sensitivity) +
                                 transpose(G_ad - G_without_covariance_sensitivity)))
interaction_norm = norm(interaction)
interaction_min_eig = minimum(eigvals(interaction))
println("COMBINED_INTERACTION,norm=", interaction_norm, ",min_eig=", interaction_min_eig)
interaction_norm > 1e-6 || error("combined FOCEI interaction term is numerically zero")
interaction_min_eig >= -1e-8 || error("expected FOCEI interaction curvature is not positive semidefinite")

G_foce = foce_curvature(subj, theta, eta)
foce_focei_gap = norm(G_ad - G_foce)
println("COMBINED_FOCE_VS_FOCEI_CURVATURE,gap=", foce_focei_gap)
foce_focei_gap > 1e-6 || error("combined-error FOCE and FOCEI curvatures are numerically identical")

_, g_fd, fd_score, fd_nconv = fd_value_grad([subj], theta; maxiter_eta=200,
                                             eps=1e-4, eps_eta=1e-4,
                                             eta_cache=nothing)
fd_imp_err = norm(g_fd - g_implicit) / max(1.0, norm(g_implicit))
println("COMBINED_FD_VS_IMPLICIT,relerr=", fd_imp_err, ",max_score=", fd_score,
        ",nconv=", fd_nconv)
fd_imp_err <= 5e-3 || error("combined FD versus implicit gradient mismatch $fd_imp_err")

termination_result = Optim.optimize(
    x -> sum(abs2, x), [1.0, -1.0], Optim.LBFGS(), Optim.Options(iterations=25))
termination_label = optimizer_termination(termination_result, EvalCache(2))
println("COMBINED_OPTIMIZER_TERMINATION,label=", termination_label)
Optim.converged(termination_result) || error("termination diagnostic smoke optimizer did not converge")
occursin("converged", termination_label) ||
    error("termination diagnostic did not record convergence")


println("FLIPFLOP_COMBINED_GRADIENT_CHECK_PASS")
