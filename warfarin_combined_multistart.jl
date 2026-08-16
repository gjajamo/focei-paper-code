# Combined additive-plus-proportional PK/PD benchmark.
ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_METHODS"] = get(ENV, "WARFARIN_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL_1NEWTON,STOP,FD")
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = "ode"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = get(ENV, "WARFARIN_JULIA_LOGDET_MODE", "raw")
ENV["WARFARIN_JULIA_DATA"] = get(ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "data", "warfarin_dat.csv"))
ENV["WARFARIN_JULIA_OUTDIR"] = get(ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "warfarin"))
include(joinpath(@__DIR__, "src", "warfarin_multistart_methods.jl"))
main()