#!/usr/bin/env julia

# Reproducible matched-start FOCEI comparison for the simulated
# Friberg--Karlsson PK--myelosuppression model. The targeted design uses 51
# subjects at 50, 80, and 110 mg; 23 PK samples through 168 h; and 77 ANC
# samples through 672 h, including a dense 6-h decline/nadir/recovery window.

const ROOT = @__DIR__

ENV["FK_DESIGN"] = get(ENV, "FK_DESIGN", "targeted_pkpd_51subjects")
ENV["FK_N_SUBJECTS"] = get(ENV, "FK_N_SUBJECTS", "51")
ENV["FK_N_STARTS"] = get(ENV, "FK_N_STARTS", "10")
ENV["FK_MAXITER_ETA"] = get(ENV, "FK_MAXITER_ETA", "50")
ENV["FK_MAXITER_OUTER"] = get(ENV, "FK_MAXITER_OUTER", "35")
ENV["FK_EBE_TOL"] = get(ENV, "FK_EBE_TOL", "6e-6")
ENV["FK_ONE_STEP_RESIDUAL_CORRECTION"] = get(
    ENV, "FK_ONE_STEP_RESIDUAL_CORRECTION", "true",
)
ENV["FK_METHODS"] = get(
    ENV,
    "FK_METHODS",
    "FULL_IMPLICIT_DIRECTIONAL_JVP,ONE_STEP_NEWTON,STOP,FD,ALMQUIST_FORWARD,LAPLACE_DIRECTIONAL_IMPLICIT",
)
ENV["FK_OUTDIR"] = get(
    ENV,
    "FK_OUTDIR",
    joinpath(ROOT, "outputs", "FribergKarlssonFOCEI", "targeted_pkpd_51subjects_matched_10"),
)

include(joinpath(ROOT, "scripts", "run_friberg_karlsson_multistart.jl"))