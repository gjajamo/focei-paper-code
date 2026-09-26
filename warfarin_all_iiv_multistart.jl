# Matched-start Warfarin FOCEI comparison with IIV on all seven structural
# parameters, including EMAX. PK and PD each use a combined additive-plus-
# proportional residual-error model.
ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_EMAX_IIV"] = "true"
ENV["WARFARIN_JULIA_METHODS"] = get(
    ENV, "WARFARIN_JULIA_METHODS",
    "FULL_IMPLICIT_DIRECTIONAL_JVP,FULL_UNROLL_1NEWTON,STOP,FD,ALMQUIST_FORWARD,LAPLACE_IMPLICIT")
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = "ode"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = get(ENV, "WARFARIN_JULIA_LOGDET_MODE", "raw")
ENV["WARFARIN_JULIA_DATA"] = get(
    ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "data", "warfarin_dat.csv"))
ENV["WARFARIN_JULIA_OUTDIR"] = get(
    ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "WarfarinCombinedAllIIV"))

include(joinpath(@__DIR__, "src", "warfarin_multistart_methods.jl"))
include(joinpath(@__DIR__, "src", "warfarin_profile_comparators.jl"))
include(joinpath(@__DIR__, "src", "almquist_warfarin_sensitivity_ode.jl"))
include(joinpath(@__DIR__, "src", "warfarin_reverse_vjp.jl"))
main()