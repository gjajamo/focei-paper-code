using LinearAlgebra
using ForwardDiff

ENV["FLIPFLOP_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["FLIPFLOP_JULIA_USE_ETA_CACHE"] = "false"
ENV["FLIPFLOP_JULIA_LOGDET_MODE"] = "raw"
ENV["FLIPFLOP_JULIA_OUTER_AD_MODE"] = "forward"

include(joinpath(@__DIR__, "..", "src", "flipflop_multistart_methods.jl"))

theta = log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.03, 0.15])
subj = simulate_subjects(theta; n_subj=1, seed=4321)[1]
eta, _, score_norm, converged = eta_mode_newton(subj, theta; maxiter=300, tol=1e-11,
                                                 backend=:forward, eta0=zeros(ETA_DIM))
converged || error("combined flip-flop EBE did not converge")

implicit_value, implicit_grad = full_implicit_subject_value_grad(subj, theta, eta)
one_step = x -> one_step_newton_subject_value(subj, x, eta)
one_step_value = one_step(theta)
one_step_grad = ForwardDiff.gradient(one_step, theta)

value_error = abs(one_step_value - implicit_value)
gradient_error = norm(one_step_grad - implicit_grad) / max(1.0, norm(implicit_grad))
println("FLIPFLOP_COMBINED_ONE_NEWTON,value_error=", value_error,
        ",gradient_relerr=", gradient_error, ",score_norm=", score_norm)

value_error <= 1e-10 || error("one-step value mismatch $value_error")
gradient_error <= 1e-9 || error("one-step versus implicit gradient mismatch $gradient_error")
score_norm <= 1e-9 || error("EBE score norm $score_norm exceeds validation threshold")

G_ad = focei_curvature(subj, theta, eta)
G_fd = focei_curvature_fd(subj, theta, eta; h=1e-5)
curvature_error = norm(G_ad - G_fd) / max(1.0, norm(G_ad))
println("FLIPFLOP_COMBINED_ONE_NEWTON_CURVATURE,relerr=", curvature_error)
curvature_error <= 5e-7 || error("combined FOCEI curvature mismatch $curvature_error")

println("FLIPFLOP_COMBINED_ONE_NEWTON_CHECK_PASS")
