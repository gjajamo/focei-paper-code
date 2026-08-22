ENV["WARFARIN_JULIA_RESIDUAL_MODEL"] = "combined"
ENV["WARFARIN_JULIA_LOGDET_MODE"] = "raw"
include(joinpath(@__DIR__, "..", "src", "warfarin_multistart_methods.jl"))
include(joinpath(@__DIR__, "..", "src", "warfarin_reverse_vjp.jl"))
subjects = parse_warfarin_csv(joinpath(@__DIR__, "..", "data", "warfarin_dat.csv"))
x = base_x0()
subj = subjects[1]
eta, _, gn, _ = eta_mode_newton(subj, x, :forward, :ode; dt=0.25, maxiter=60, tol=1e-9)
oldv, oldg = full_implicit_subject_value_grad(subj, x, eta, :ode; dt=0.25)
newv, newg = full_implicit_reverse_vjp_subject_value_grad(subj, x, eta, :ode; dt=0.25)
gold = ForwardDiff.gradient(e -> h_i(subj,x,e,:ode;dt=0.25), eta)
Hold = ForwardDiff.hessian(e -> h_i(subj,x,e,:ode;dt=0.25), eta)
gnew = reverse_warfarin_score(subj,x,eta,:ode;dt=0.25)
Hnew = ReverseDiff.jacobian(e -> reverse_warfarin_score(subj,x,e,:ode;dt=0.25),eta)
println("eta=", eta, " score_norm=", gn)
println("value old/new/diff=", oldv, " ", newv, " ", newv-oldv)
println("grad relerr=", norm(newg-oldg)/max(norm(oldg),1.0))
println("score relerr=", norm(gnew-gold)/max(norm(gold),1.0))
println("hess relerr=", norm(Hnew-Hold)/max(norm(Hold),1.0))