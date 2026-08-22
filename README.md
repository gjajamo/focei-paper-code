# FOCEI paper code

Reproducible Julia implementation for the combined additive-plus-proportional residual-error benchmarks reported in the FOCEI automatic-differentiation manuscript:

- a simulated one-compartment population PK model with absorption/elimination ambiguity;
- a public warfarin PK/PD model with separate combined PK and PD residual-error models.

## Methods

The primary manuscript comparison uses matched starts for four population-gradient strategies:

- `FULL_IMPLICIT`: implicit/adjoint AD through the converged EBE score equation;
- `FULL_UNROLL_1NEWTON`: one exact Newton update from a detached converged EBE;
- `STOP`: AD with the converged EBE held fixed;
- `FD`: fully finite-difference population profiling and EBE determination.

`FULL_UNROLL_1NEWTON` is the code name for the manuscript's one-step Newton method.

The release also contains a head-to-head implementation comparison of two mathematically equivalent full implicit derivatives:

- `ALMQUIST_FORWARD`: forward sensitivities of the EBE score equation, following Almquist et al.;
- `FULL_IMPLICIT_DIRECTIONAL_JVP`: the adjoint form with an exact one-direction EBE Jacobian--vector product (JVP).

For a subject conditional objective `h(theta, eta)`, score `g = d h / d eta`, mixed score derivative `B = d g / d theta`, and adjoint `lambda`, the directional implementation evaluates

```text
B' * lambda = d/dtheta [ d/dt h(theta, eta + t * lambda) | t = 0 ].
```

It therefore obtains precisely the mixed contraction needed by the implicit FOCEI gradient without materializing the full EBE-sensitivity matrix. It is not a different FOCEI approximation or a one-step-Newton method.

With identical combined-error targets, exact-Newton EBE solvers, starts, iteration limits, and eight Julia threads, its paired median wall-time speed-up versus `ALMQUIST_FORWARD` was 1.04-fold for the 100-start synthetic PK comparison and 1.07-fold for the 10-start warfarin comparison. Local gradient agreement was at machine precision.

## Requirements

- Julia 1.10.4 (the committed `Manifest.toml` locks the manuscript environment);
- the Julia packages in `Project.toml`;
- eight CPU threads for the reported multi-start runs.

Instantiate the environment once:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Data

The one-compartment data are simulated deterministically by the code (seed 123). The warfarin data are public but are not redistributed here. Download them before running the warfarin benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_warfarin_data.ps1
```

The download is saved to `data/warfarin_dat.csv`, which is ignored by Git. The source data and workshop materials are:

- Holford N. *Warfarin PK/PD workshop materials and dataset (public).* University of Auckland. [Workshop materials](https://holford.fmhs.auckland.ac.nz/docs/PKPDWorkshop/WarfarinUnderstanding.pdf); [dataset](https://holford.fmhs.auckland.ac.nz/research/nlmixr/warfarin/warfarin_dat.csv). Accessed March 12, 2026.

## Run the primary benchmarks

```powershell
julia -t 8 --project=. flipflop_combined_multistart.jl
julia -t 8 --project=. warfarin_combined_multistart.jl
```

The default runs use 100 matched starts for the one-compartment model and 10 for warfarin. Results are written under `outputs/`, which is ignored by Git.

## Run the directional-JVP versus forward-sensitivity comparison

After downloading the warfarin data, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_directional_jvp_comparison.ps1 -Threads 8
```

The script uses the same generated start bank for `ALMQUIST_FORWARD` and `FULL_IMPLICIT_DIRECTIONAL_JVP` within each case study. It writes its results under `outputs/DirectionalJVPvsAlmquist/`.

For a short smoke run, lower the run limits before invoking a wrapper:

```powershell
$env:FLIPFLOP_JULIA_N_STARTS = '1'
$env:FLIPFLOP_JULIA_MAXITER_OUTER = '2'
$env:FLIPFLOP_JULIA_MAXITER_ETA = '3'
$env:FLIPFLOP_JULIA_METHODS = 'ALMQUIST_FORWARD,FULL_IMPLICIT_DIRECTIONAL_JVP'
julia -t 2 --project=. flipflop_combined_multistart.jl
```

## Validation

The `test/` directory contains focused checks for the combined-error gradient, one-step Newton construction, and reverse-VJP prototype. The public implementation is research code. Before interpreting a full run, inspect EBE convergence diagnostics, common endpoint evaluations, and method-specific optimizer termination fields.

## Reproducibility scope

The source package contains the implementations needed to reproduce the reported combined-error case studies. It intentionally excludes manuscript figures, private workspace files, historical additive-only analyses, and output bundles. The repository license states the applicable reuse terms.