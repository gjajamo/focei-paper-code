#!/usr/bin/env julia

# Local derivative validation for the exploratory Warfarin model with IIV on
# all seven structural parameters, including EMAX.
ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_EMAX_IIV"] = "true"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = "raw"

root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(root, "src", "warfarin_multistart_methods.jl"))
include(joinpath(root, "src", "warfarin_profile_comparators.jl"))

subjects = parse_warfarin_csv(joinpath(root, "data", "warfarin_dat.csv"))[1:2]
x = base_x0()
length(x) == 18 || error("expected 18 population parameters, found $(length(x))")
etas, max_score, n_converged = solve_all_etas(subjects, x, :ode; maxiter=30, backend=:forward)
value_forward, grad_forward = almquist_forward_value_grad(
    subjects, x, :ode; maxiter_eta=30
)
value_directional, grad_directional, _, _ = full_implicit_directional_jvp_value_grad(
    subjects, x, :ode; maxiter_eta=30
)
scale = max(1.0, norm(grad_forward), norm(grad_directional))
println("eta_dim=$(ETA_DIM) parameter_dim=$(length(x)) n_subjects=$(length(subjects))")
println("independent_mode_check=max_score=$(max_score) converged=$(n_converged)/$(length(subjects))")
println("value_abs_difference=$(abs(value_forward - value_directional))")
println("gradient_relative_difference=$(norm(grad_forward - grad_directional) / scale)")
