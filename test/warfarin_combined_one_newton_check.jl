using LinearAlgebra
using ForwardDiff

ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = "raw"
ENV["WARFARIN_JULIA_ETA_POLISH_MAXITER"] = "12"

include(joinpath(@__DIR__, "..", "src", "warfarin_multistart_methods.jl"))

subjects = parse_warfarin_csv(joinpath(@__DIR__, "..", "data", "warfarin_dat.csv"))
subj = subjects[1]
x = base_x0()
@assert length(x) == 17
@assert length(PARAM_NAMES) == 17
@assert length(theta_bounds()[1]) == 17

eta, _, score_norm, converged = eta_mode_newton_focei(
    subj, x, :ode; maxiter=80, tol=1e-9)
converged || error("combined-error EBE did not converge: score norm=$score_norm")

value_implicit, grad_implicit = full_implicit_subject_value_grad(subj, x, eta, :ode)
one_step = xx -> one_step_newton_subject_value(subj, xx, eta, :ode)
value_one_step = one_step(x)
grad_one_step = ForwardDiff.gradient(one_step, x)

value_error = abs(value_one_step - value_implicit)
gradient_error = norm(grad_one_step - grad_implicit) / max(1.0, norm(grad_implicit))
println("COMBINED_ONE_NEWTON,value_error=", value_error,
        ",gradient_relerr=", gradient_error, ",score_norm=", score_norm)
value_error <= 1e-8 || error("one-step value mismatch $value_error")
gradient_error <= 2e-6 || error("one-step gradient mismatch $gradient_error")

G_ad = focei_curvature(subj, x, eta, :ode)
G_fd = focei_curvature_fd(subj, x, eta, :ode; h=1e-5)
curvature_error = norm(G_ad - G_fd) / max(1.0, norm(G_ad))
println("COMBINED_CURVATURE,relerr=", curvature_error)
curvature_error <= 2e-6 || error("combined FOCEI curvature mismatch $curvature_error")

println("WARFARIN_COMBINED_ONE_NEWTON_CHECK_PASS")
