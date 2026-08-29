#!/usr/bin/env julia

# Reproducible matched-start FOCEI comparison for the public warfarin PK/PD
# data. This runner puts IIV on all seven structural parameters and uses
# separate combined additive-plus-proportional residual-error models for PK
# and PD. Every method uses the same deterministic start bank.

const ROOT = normpath(joinpath(@__DIR__, ".."))

ENV["WARFARIN_JULIA_N_SUBJ"] = get(ENV, "WARFARIN_JULIA_N_SUBJ", "32")
ENV["WARFARIN_JULIA_N_STARTS"] = get(ENV, "WARFARIN_JULIA_N_STARTS", "10")
ENV["WARFARIN_JULIA_MAXITER_ETA"] = get(ENV, "WARFARIN_JULIA_MAXITER_ETA", "30")
ENV["WARFARIN_JULIA_MAXITER_OUTER"] = get(ENV, "WARFARIN_JULIA_MAXITER_OUTER", "50")
ENV["WARFARIN_JULIA_METHODS"] = get(
    ENV,
    "WARFARIN_JULIA_METHODS",
    "FULL_IMPLICIT_DIRECTIONAL_JVP,FULL_UNROLL_1NEWTON,STOP,FD,ALMQUIST_FORWARD,LAPLACE_IMPLICIT",
)
ENV["WARFARIN_JULIA_REPRESENTATIONS"] = get(ENV, "WARFARIN_JULIA_REPRESENTATIONS", "ode")
ENV["WARFARIN_JULIA_DATA"] = get(
    ENV,
    "WARFARIN_JULIA_DATA",
    joinpath(ROOT, "data", "warfarin_dat.csv"),
)
ENV["WARFARIN_JULIA_OUTDIR"] = get(
    ENV,
    "WARFARIN_JULIA_OUTDIR",
    joinpath(ROOT, "outputs", "WarfarinCombinedAllIIV", "matched_10"),
)

include(joinpath(ROOT, "warfarin_all_iiv_multistart.jl"))