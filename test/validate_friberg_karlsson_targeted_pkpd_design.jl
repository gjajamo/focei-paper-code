#!/usr/bin/env julia

# Smoke test for the targeted Friberg--Karlsson PK--PD observation design.
const ROOT = normpath(joinpath(@__DIR__, ".."))
ENV["FK_DESIGN"] = "targeted_pkpd_51subjects"
ENV["FK_EBE_TOL"] = "6e-6"
include(joinpath(ROOT, "scripts", "run_friberg_karlsson_scale_benchmark.jl"))

function main()
    truth = true_theta()
    subjects = simulate_population(MersenneTwister(20260822), truth)
    expected_pk = Float64[0.083, 0.167, 0.25, 0.333, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6,
                          8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168]
    expected_anc = sort(unique(vcat(collect(0.0:12.0:672.0), collect(96.0:6.0:336.0))))
    @assert FK_DESIGN == "targeted_pkpd_51subjects"
    @assert length(subjects) == 51
    @assert PK_TIMES == expected_pk
    @assert ANC_TIMES == expected_anc
    @assert length(PK_TIMES) == 23 && length(ANC_TIMES) == 77
    @assert all(count(==(dose), getfield.(subjects, :dose)) == 17 for dose in FK_DOSE_LEVELS)
    @assert all(length(subj.times) == 100 && all(isfinite, subj.y) && all(subj.y .> 0) for subj in subjects)
    etas, max_score, nconverged = solve_all_modes(subjects, truth)
    @assert nconverged == length(subjects)
    @assert max_score <= parse(Float64, ENV["FK_EBE_TOL"])
    @assert all(all(isfinite, eta) for eta in etas)
    println("FRIBERG_KARLSSON_TARGETED_PKPD_DESIGN_PASS")
    println("subjects=$(length(subjects)) doses_mg=$(join(FK_DOSE_LEVELS_MG, ';')) pk_observations=$(length(PK_TIMES)) anc_observations=$(length(ANC_TIMES)) total_per_subject=$(length(subjects[1].times))")
    println("anc_fine_window_h=96-336 max_ebe_score=$max_score n_ebe_converged=$nconverged")
end
main()