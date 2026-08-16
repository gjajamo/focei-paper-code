# FOCEI paper code

Reproducible Julia implementation for the two combined additive-plus-proportional residual-error benchmarks reported in the FOCEI automatic-differentiation manuscript:

- a simulated one-compartment population PK model with absorption/elimination ambiguity;
- a public warfarin PK/PD model with separate combined PK and PD residual-error models.

The release compares four population-gradient methods using matched starting points:

- FULL_IMPLICIT - implicit/adjoint AD through the converged EBE score equation;
- FULL_UNROLL_1NEWTON - AD through one exact Newton update from a detached converged EBE;
- STOP - AD with the converged EBE held fixed;
- FD - fully finite-difference population and EBE profiling.

FULL_UNROLL_1NEWTON is the code name for the one-step Newton method in the manuscript.

## Requirements

- Julia 1.10.4 (the committed Manifest.toml locks the manuscript environment);
- Julia packages in Project.toml;
- eight CPU threads for the reported multi-start runs.

Instantiate the environment once:

~~~powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
~~~

## Data

The one-compartment data are simulated deterministically by the code (seed 123). The warfarin data are public but are not redistributed here. Download them before running the warfarin benchmark:

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts/download_warfarin_data.ps1
~~~

The download is saved to data/warfarin_dat.csv, which is ignored by Git. The source data and workshop materials are:

- Holford N. *Warfarin PK/PD workshop materials and dataset (public).* University of Auckland.
  [Workshop materials](https://holford.fmhs.auckland.ac.nz/docs/PKPDWorkshop/WarfarinUnderstanding.pdf);
  [dataset](https://holford.fmhs.auckland.ac.nz/research/nlmixr/warfarin/warfarin_dat.csv).
  Accessed March 12, 2026.

## Run the benchmarks

~~~powershell
julia -t 8 --project=. flipflop_combined_multistart.jl
julia -t 8 --project=. warfarin_combined_multistart.jl
~~~

The default runs use 100 matched starts for the one-compartment model and 10 for warfarin. Results are written under outputs/, which is ignored by Git.

For a short smoke run, set lower run limits before invoking the corresponding script:

~~~powershell
$env:FLIPFLOP_JULIA_N_STARTS = '1'
$env:FLIPFLOP_JULIA_MAXITER_OUTER = '2'
$env:FLIPFLOP_JULIA_MAXITER_ETA = '3'
julia -t 2 --project=. flipflop_combined_multistart.jl
~~~

## Reproducibility scope

The source package intentionally contains the implementations necessary to reproduce the reported combined-error case studies. It does not include manuscript figures, private workspace files, historical additive-only analyses, or output bundles. The repository’s license states the applicable reuse terms.