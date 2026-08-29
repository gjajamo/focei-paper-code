# FOCEI derivative benchmark code

This repository contains the Julia code for the FOCEI automatic-differentiation manuscript. It reproduces the two combined additive-plus-proportional residual-error benchmarks used in the paper:

- a deterministic simulated Friberg--Karlsson PK--myelosuppression model;
- a public warfarin PK/PD model.

The repository intentionally excludes manuscript files, generated figures, simulation outputs, private workspace material, and the superseded one-compartment exploratory case study.

## Benchmark models

The Friberg--Karlsson benchmark has eight ODE states, nine structural population parameters, seven IIV parameters, and four residual-error parameters (20 population parameters total). Its retained targeted design has 51 subjects assigned equally to 50-, 80-, and 110-mg oral doses. Each subject has 23 PK observations through 168 h and 77 ANC observations through 672 h, with additional 6-h ANC samples across the decline--nadir--recovery interval.

The warfarin benchmark uses the public PK/PD dataset, seven structural parameters with IIV, and separate combined error models for PK and PD (18 population parameters total). The supplied runner uses the first 32 subjects, as in the study's matched-start comparison.

## Methods

The scripts expose the following population-derivative strategies:

- `FULL_IMPLICIT_DIRECTIONAL_JVP`: the implicit/adjoint FOCEI derivative, computing the required mixed-derivative contraction without materializing the complete EBE-sensitivity matrix;
- `ONE_STEP_NEWTON` (Friberg--Karlsson) and `FULL_UNROLL_1NEWTON` (warfarin): one exact Newton update from a detached converged EBE; the Friberg--Karlsson runner enables its finite-residual correction by default;
- `STOP`: AD with the EBE treated as fixed;
- `FD`: an independently profiled finite-difference baseline;
- `ALMQUIST_FORWARD`: the forward EBE-sensitivity calculation following Almquist et al.;
- `LAPLACE_DIRECTIONAL_IMPLICIT` / `LAPLACE_IMPLICIT`: a directional implicit Laplace comparator. Its objective differs from the FOCEI target, so it is useful for timing and implementation comparisons rather than direct FOCEI objective comparisons.

For a conditional objective `h(theta, eta)`, score `g = d h / d eta`, mixed score derivative `B = d g / d theta`, and adjoint `lambda`, the directional implementation evaluates the required contraction as

```text
B' * lambda = d/dtheta [ d/dt h(theta, eta + t * lambda) | t = 0 ].
```

## Requirements

- Julia 1.10.4 (the committed `Manifest.toml` fixes the study environment);
- the packages declared in `Project.toml`;
- multiple CPU threads for practical multi-start runs. The manuscript timing experiments used eight Julia threads.

Instantiate the project once:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Warfarin data

The warfarin data are public but are not redistributed here. Download them before running the warfarin benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_warfarin_data.ps1
```

This writes `data/warfarin_dat.csv`, which is ignored by Git. The source data and workshop materials are:

- Holford N. *Warfarin PK/PD workshop materials and dataset (public).* University of Auckland. [Workshop materials](https://holford.fmhs.auckland.ac.nz/docs/PKPDWorkshop/WarfarinUnderstanding.pdf); [dataset](https://holford.fmhs.auckland.ac.nz/research/nlmixr/warfarin/warfarin_dat.csv). Accessed March 12, 2026.

## Run the matched-start benchmarks

The top-level Friberg--Karlsson runner defaults to ten matched starts, the targeted 51-subject design, and all six methods:

```powershell
julia -t 8 --project=. friberg_karlsson_targeted_multistart.jl
```

After downloading the data, run the all-IIV warfarin comparison:

```powershell
julia -t 8 --project=. scripts/run_warfarin_all_iiv_matched.jl
```

Both runners write deterministic start banks and result tables under `outputs/`, which is ignored by Git. Run one method at a time with the same start bank by setting the applicable `*_METHODS` and `*_OUTDIR` environment variables. For example:

```powershell
$env:FK_METHODS = 'FULL_IMPLICIT_DIRECTIONAL_JVP,ALMQUIST_FORWARD'
$env:FK_N_STARTS = '2'
julia -t 8 --project=. friberg_karlsson_targeted_multistart.jl
```

## Validation

The targeted Friberg--Karlsson design and two local warfarin derivative checks are provided in `test/`. They are intended as focused checks, not as short unit tests: each performs ODE solves and EBE optimizations.

```powershell
julia -t 8 --project=. test/validate_friberg_karlsson_targeted_pkpd_design.jl
julia -t 8 --project=. test/validate_warfarin_all_iiv.jl
julia -t 8 --project=. test/validate_warfarin_one_step_explicit.jl
```

Before interpreting a full optimization run, inspect the recorded EBE convergence diagnostics, common endpoint evaluations, and optimizer termination fields. This is research code supplied to reproduce the study.