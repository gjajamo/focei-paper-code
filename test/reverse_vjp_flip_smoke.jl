ENV["FLIPFLOP_JULIA_RESIDUAL_MODEL"] = "combined"
include(joinpath(@__DIR__, "..", "src", "flipflop_multistart_methods.jl"))
include(joinpath(@__DIR__, "..", "src", "flipflop_reverse_vjp.jl"))
theta = log.([0.05, 3.0, 20.0, 0.1, 0.1, 0.03, 0.15])
subj = simulate_subjects(theta; n_subj=1, seed=123)[1]
eta, _, gn, _ = eta_mode_newton(subj, theta; maxiter=60, tol=1e-10)
oldv, oldg = full_implicit_subject_value_grad(subj, theta, eta)
newv, newg = full_implicit_reverse_vjp_subject_value_grad(subj, theta, eta)
println("eta=", eta, " score_norm=", gn)
println("value old/new/diff=", oldv, " ", newv, " ", newv-oldv)
println("grad relerr=", norm(newg-oldg)/max(norm(oldg),1.0))
println("score relerr=", norm(reverse_subject_score(subj,theta,eta)-ForwardDiff.gradient(e->subject_nll(subj,theta,e),eta)))
println("hess relerr=", norm(ReverseDiff.jacobian(e->reverse_subject_score(subj,theta,e),eta)-ForwardDiff.hessian(e->subject_nll(subj,theta,e),eta))/max(norm(ForwardDiff.hessian(e->subject_nll(subj,theta,e),eta)),1.0))