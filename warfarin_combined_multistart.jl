ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_METHODS"] = get(
    ENV, "WARFARIN_JULIA_METHODS", "FULL_IMPLICIT,FULL_UNROLL_1NEWTON,FULL_UNROLL,STOP,FD")
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = "ode"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = get(ENV, "WARFARIN_JULIA_LOGDET_MODE", "raw")
ENV["WARFARIN_JULIA_FULL_UNROLL_USE_ETA_CACHE"] = "false"
ENV["WARFARIN_JULIA_FULL_UNROLL_AUTO_RETRY"] = "false"
ENV["WARFARIN_JULIA_DATA"] = get(
    ENV, "WARFARIN_JULIA_DATA", joinpath(@__DIR__, "data", "warfarin_dat.csv"))
ENV["WARFARIN_JULIA_OUTDIR"] = get(
    ENV, "WARFARIN_JULIA_OUTDIR", joinpath(@__DIR__, "outputs", "WarfarinCombinedFOCEIMultistart"))

include(joinpath(@__DIR__, "src", "warfarin_multistart_methods.jl"))
include(joinpath(@__DIR__, "src", "warfarin_profile_comparators.jl"))
include(joinpath(@__DIR__, "src", "warfarin_reverse_vjp.jl"))
main()
