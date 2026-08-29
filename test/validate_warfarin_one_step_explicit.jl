#!/usr/bin/env julia

# Validate that the explicit residual-corrected tangent is the derivative of
# the same detached one-Newton map as the retained direct-AD reference.  This
# is deliberately one subject so it can run between case-study queues without
# adding meaningful wall time to the comparison.

const ROOT = normpath(joinpath(@__DIR__, ".."))
ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_EMAX_IIV"] = "true"
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = "ode"
ENV["WARFARIN_JULIA_DATA"] = joinpath(ROOT, "data", "warfarin_dat.csv")

include(joinpath(ROOT, "src", "warfarin_multistart_methods.jl"))

function compare_one_step(subject, x, eta, label)
    reference = xx -> one_step_newton_subject_value_ad_reference(subject, xx, eta, :ode)
    value_reference = Float64(reference(x))
    gradient_reference = Vector{Float64}(ForwardDiff.gradient(reference, x))
    value_explicit, gradient_explicit = one_step_newton_subject_value_grad_explicit(subject, x, eta, :ode)
    relative_value = abs(value_explicit - value_reference) / max(1.0, abs(value_reference))
    relative_gradient = norm(gradient_explicit - gradient_reference) /
                        max(1.0, norm(gradient_reference))
    println("$label value_relative_error=$relative_value gradient_relative_error=$relative_gradient")
    return relative_value, relative_gradient
end

subjects = parse_warfarin_csv(ENV["WARFARIN_JULIA_DATA"])
subject = subjects[1]
x = base_x0()
eta, _, score_norm, converged = eta_mode_newton_focei(subject, x, :ode; maxiter=30)
converged || error("validation EBE did not converge; score norm=$score_norm")

root_value_error, root_gradient_error = compare_one_step(subject, x, eta, "converged_EBE")
residual_eta = eta .+ 0.01
residual_value_error, residual_gradient_error = compare_one_step(
    subject, x, residual_eta, "finite_residual_EBE",
)

tolerance = 1.0e-7
maximum((root_value_error, root_gradient_error, residual_value_error, residual_gradient_error)) <= tolerance ||
    error("explicit one-step validation failed at tolerance $tolerance")

println("WARFARIN_EXPLICIT_ONE_STEP_VALIDATION_PASS")
