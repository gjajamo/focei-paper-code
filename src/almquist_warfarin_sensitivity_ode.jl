# Literal Almquist-style sensitivity-ODE implementation for the combined-error,
# all-IIV warfarin FOCEI case study. Runtime calculations are Float64 only.
include(joinpath(@__DIR__, "generated_warfarin_sensitivity_derivatives.jl"))

const WARF_SENS_ODE_MAP = (1, 2, 3, 6, 8, 9, 10, 13)
const WARF_SENS_PAIRS = vcat([(a,b) for a in 8:14 for b in a:14], [(a,b) for a in 8:14 for b in 1:7])
@inline warf_d(x) = length(x) + ETA_DIM
@inline warf_erange(x) = (length(x)+1):(length(x)+ETA_DIM)
function warf_check(x, eta)
    warfarin_combined_error() || error("requires combined error")
    ETA_DIM == 7 || error("requires all-IIV warfarin")
    length(x) == 18 && length(eta) == 7 || error("unexpected warfarin dimensions")
end
function warf_z(x, eta)
    warf_check(x, eta); vcat(x[1:7], eta)
end
function warf_coeff(t, dose, ce, z)
    r = warfarin_ce_rhs_sensitivity_derivatives(t, dose, ce, z[collect(WARF_SENS_ODE_MAP)])
    cp,cpw,cpww,f,fc,fw,fcw,fww = r
    a=zeros(14); b=zeros(14); c=zeros(14,14); d=zeros(14); e=zeros(14,14)
    for i in 1:8
        ii=WARF_SENS_ODE_MAP[i]; a[ii]=fw[i]; b[ii]=fcw[i]; d[ii]=cpw[i]
        for j in 1:8
            jj=WARF_SENS_ODE_MAP[j]; c[ii,jj]=fww[i,j]; e[ii,jj]=cpww[i,j]
        end
    end
    return cp,d,e,f,fc,a,b,c
end
function warf_rhs(t,dose,ce,S,T,z)
    _,_,_,f,fc,fw,fcw,fww=warf_coeff(t,dose,ce,z)
    ds=fc.*S .+ fw; dT=zeros(14,14)
    for (a,b) in WARF_SENS_PAIRS
        v=fc*T[a,b]+fcw[b]*S[a]+fcw[a]*S[b]+fww[a,b]
        dT[a,b]=v; dT[b,a]=v
    end
    f,ds,dT
end
function warf_step(t,dose,ce,S,T,h,z)
    k1,k1s,k1t=warf_rhs(t,dose,ce,S,T,z)
    k2,k2s,k2t=warf_rhs(t+h/2,dose,ce+h*k1/2,S.+h.*k1s./2,T.+h.*k1t./2,z)
    k3,k3s,k3t=warf_rhs(t+h/2,dose,ce+h*k2/2,S.+h.*k2s./2,T.+h.*k2t./2,z)
    k4,k4s,k4t=warf_rhs(t+h,dose,ce+h*k3,S.+h.*k3s,T.+h.*k3t,z)
    ce+h*(k1+2*k2+2*k3+k4)/6, S.+h.*(k1s.+2 .*k2s.+2 .*k3s.+k4s)./6, T.+h.*(k1t.+2 .*k2t.+2 .*k3t.+k4t)./6
end
function warf_ce_traj(times,dose,z;dt=0.25)
    ce=0.; S=zeros(14); T=zeros(14,14); now=0.; out=Vector{Tuple{Float64,Vector{Float64},Matrix{Float64}}}(undef,length(times))
    for i in eachindex(times)
        n=max(1,ceil(Int,max(times[i]-now,0)/dt)); h=max(times[i]-now,0)/n
        for _ in 1:n; ce,S,T=warf_step(now,dose,ce,S,T,h,z); now+=h; end
        out[i]=(ce,copy(S),copy(T))
    end
    out
end
function warf_expand(value,ds,d2s,x)
    D=warf_d(x); Pp=length(x); d=zeros(D); d2=zeros(D,D)
    for a in 1:14
        ia=a<=7 ? a : Pp+a-7; d[ia]=ds[a]
        for b in 1:14; ib=b<=7 ? b : Pp+b-7; d2[ia,ib]=d2s[a,b]; end
    end
    value,d,d2
end
function warf_predictions(subj,x,eta;dt=0.25)
    z=warf_z(x,eta); n=length(subj.pk_obs)+length(subj.pd_obs)
    mu=Vector{Float64}(undef,n); d=Vector{Vector{Float64}}(undef,n); d2=Vector{Matrix{Float64}}(undef,n); endpoint=Vector{Int}(undef,n); y=Vector{Float64}(undef,n)
    for i in eachindex(subj.pk_obs)
        cp,cpw,cpww,_,_,_,_,_=warf_coeff(subj.pk_times[i],subj.dose_mg,0.,z)
        mu[i],d[i],d2[i]=warf_expand(cp,cpw,cpww,x); endpoint[i]=1; y[i]=subj.pk_obs[i]
    end
    out=warf_ce_traj(subj.pd_times,subj.dose_mg,z;dt=dt); off=length(subj.pk_obs)
    for i in eachindex(subj.pd_obs)
        ce,S,T=out[i]; v,dc,dw,dcc,dcw,dww=warfarin_pd_output_derivatives(ce,z); gp=dc.*S.+dw; hp=zeros(14,14)
        for a in 1:14,b in 1:14; hp[a,b]=dcc*S[a]*S[b]+dc*T[a,b]+dcw[a]*S[b]+dcw[b]*S[a]+dww[a,b]; end
        mu[off+i],d[off+i],d2[off+i]=warf_expand(v,gp,hp,x); endpoint[off+i]=2; y[off+i]=subj.pd_obs[i]
    end
    mu,d,d2,endpoint,y
end
@inline warf_ridx(k)=k==1 ? (8,10) : (9,11)
function warf_vder(mu,d,d2,x,endpoint)
    ai,pi=warf_ridx(endpoint); D=warf_d(x); a2=exp(2*x[ai]); p2=exp(2*x[pi]); v=a2+p2*mu^2; dv=zeros(D); d2v=zeros(D,D)
    for a in 1:D
        da=a==ai ? 1. : 0.; pa=a==pi ? 1. : 0.; dv[a]=2*a2*da+2*p2*pa*mu^2+2*p2*mu*d[a]
        for b in 1:D
            db=b==ai ? 1. : 0.; pb=b==pi ? 1. : 0.
            d2v[a,b]=4*a2*da*db+4*p2*pa*pb*mu^2+4*p2*mu*(pa*d[b]+pb*d[a])+2*p2*(d[a]*d[b]+mu*d2[a,b])
        end
    end
    v,dv,d2v,p2
end
function warf_addobs!(href,grad,H,y,mu,d,d2,x,endpoint)
    v,dv,d2v,_=warf_vder(mu,d,d2,x,endpoint); r=y-mu; href[]+=.5*(log(v)+r^2/v)
    pv=.5/v-.5*r^2/v^2; pr=r/v; pvv=-.5/v^2+r^2/v^3; pvr=-r/v^2; prr=1/v
    for a in eachindex(grad)
        ra=-d[a]; grad[a]+=pv*dv[a]+pr*ra
        for b in eachindex(grad)
            rb=-d[b]; rab=-d2[a,b]
            H[a,b]+=pvv*dv[a]*dv[b]+pvr*(dv[a]*rb+ra*dv[b])+prr*ra*rb+pv*d2v[a,b]+pr*rab
        end
    end
end
function warf_prior!(href,grad,H,x,eta)
    Pp=length(x)
    for j in 1:ETA_DIM
        ti=11+j; ei=Pp+j; w=exp(x[ti]); iw2=inv(w^2); href[]+=log(w)+.5*eta[j]^2*iw2
        grad[ti]+=1-eta[j]^2*iw2; grad[ei]+=eta[j]*iw2; H[ti,ti]+=2*eta[j]^2*iw2
        H[ti,ei]+=-2*eta[j]*iw2; H[ei,ti]+=-2*eta[j]*iw2; H[ei,ei]+=iw2
    end
end
function warf_G_dld(x,mu,d,d2,endpoint)
    D=warf_d(x); Pp=length(x); er=warf_erange(x); G=zeros(ETA_DIM,ETA_DIM); dG=[zeros(ETA_DIM,ETA_DIM) for _ in 1:D]
    for i in eachindex(mu)
        v,dv,_,p2=warf_vder(mu[i],d[i],d2[i],x,endpoint[i]); J=view(d[i],er); V=2 .*p2 .*mu[i] .*J; pi=warf_ridx(endpoint[i])[2]
        for a in 1:ETA_DIM,b in 1:ETA_DIM; G[a,b]+=J[a]*J[b]/v+.5*V[a]*V[b]/v^2; end
        for z in 1:D,a in 1:ETA_DIM,b in 1:ETA_DIM
            Jaz=d2[i][Pp+a,z]; Jbz=d2[i][Pp+b,z]
            Vaz=2*p2*(d[i][z]*J[a]+mu[i]*Jaz)+(z==pi ? 4*p2*mu[i]*J[a] : 0.)
            Vbz=2*p2*(d[i][z]*J[b]+mu[i]*Jbz)+(z==pi ? 4*p2*mu[i]*J[b] : 0.)
            dG[z][a,b]+=-dv[z]*J[a]*J[b]/v^2+(Jaz*J[b]+J[a]*Jbz)/v-V[a]*V[b]*dv[z]/v^3+.5*(Vaz*V[b]+V[a]*Vbz)/v^2
        end
    end
    for j in 1:ETA_DIM
        ti=11+j; iw2=exp(-2*x[ti]); G[j,j]+=iw2; dG[ti][j,j]+=-2*iw2
    end
    F=cholesky(Symmetric((G+transpose(G))/2);check=true); Ginv=inv(F); ld=2sum(log,diag(F.L)); dld=[sum(Ginv .*dG[z]) for z in 1:D]
    G,ld,dld
end
function warf_components(subj,x,eta;dt=.25)
    mu,d,d2,endpoint,y=warf_predictions(subj,x,eta;dt=dt); D=warf_d(x); h=Ref(0.); g=zeros(D); H=zeros(D,D)
    for i in eachindex(mu); warf_addobs!(h,g,H,y[i],mu[i],d[i],d2[i],x,endpoint[i]); end
    warf_prior!(h,g,H,x,eta); G,ld,dld=warf_G_dld(x,mu,d,d2,endpoint)
    h[],g,H,G,ld,dld
end
function warf_solve_mode(subj,x,representation;dt=.25,maxiter=30,tol=1e-7,eta0=zeros(ETA_DIM))
    representation==:ode || error("supports only :ode"); eta=Float64.(eta0); er=warf_erange(x)
    for _ in 1:maxiter
        f,gall,_,G,_,_=warf_components(subj,x,eta;dt=dt); g=gall[er]
        (!isfinite(f)||any(!isfinite,g)||norm(g)<tol) && break
        step=safe_solve(G,g); alpha=1.; accepted=false
        for _ in 1:14
            trial=eta.-alpha.*step; ft=warf_components(subj,x,trial;dt=dt)[1]
            if strict_descent_accept(ft,f,g,step,alpha); eta=trial; accepted=true; break; end
            alpha*=.5
        end
        accepted||break
    end
    # Match the existing FOCEI EBE routine: after working-curvature iterations,
    # polish the residual score with the analytic exact conditional Hessian.
    score = norm(warf_components(subj, x, eta; dt=dt)[2][er])
    if isfinite(score) && score >= tol
        for _ in 1:8
            f, gall, Hall, _, _, _ = warf_components(subj, x, eta; dt=dt)
            g = gall[er]
            score = norm(g)
            score < tol && break
            Heta = (Hall[er, er] + transpose(Hall[er, er])) / 2
            step = safe_solve(Heta, g)
            alpha = 1.0
            accepted = false
            for _ in 1:14
                trial = eta .- alpha .* step
                ftry = warf_components(subj, x, trial; dt=dt)[1]
                if strict_descent_accept(ftry, f, g, step, alpha)
                    eta = trial
                    accepted = true
                    break
                end
                alpha *= 0.5
            end
            accepted || break
        end
    end
    score = norm(warf_components(subj, x, eta; dt=dt)[2][er])
    eta, score, isfinite(score) && score < tol
end
function warf_solve_all(subjects,x,representation;dt=.25,maxiter=30,eta_cache=nothing)
    etas=Vector{Vector{Float64}}(undef,length(subjects)); score=zeros(length(subjects)); cvg=Vector{Bool}(undef,length(subjects))
    @threads for i in eachindex(subjects)
        eta,s,c=warf_solve_mode(subjects[i],x,representation;dt=dt,maxiter=maxiter,eta0=eta_start(eta_cache,subjects[i])); etas[i]=eta; score[i]=s; cvg[i]=c
    end
    etas,maximum(score),count(cvg)
end
function almquist_sensitivity_ode_subject_value_grad(subj,x,eta,representation;dt=.25)
    h,g,H,_,ld,dld=warf_components(subj,x,eta;dt=dt); er=warf_erange(x); Pp=length(x); Heta=(H[er,er]+transpose(H[er,er]))/2; S=-(cholesky(Symmetric(Heta);check=true) \ Matrix{Float64}(H[er,1:Pp]))
    value=2*(h+.5ld); grad=2 .* (g[1:Pp].+.5 .*dld[1:Pp].+transpose(S)*(.5 .*dld[er]))
    value,Vector{Float64}(grad)
end
function almquist_sensitivity_ode_value_grad(subjects,x,representation;dt=.25,maxiter_eta=30,eta_cache=nothing)
    etas,maxs,nc=warf_solve_all(subjects,x,representation;dt=dt,maxiter=maxiter_eta,eta_cache=eta_cache)
    vals=zeros(length(subjects)); grads=[zeros(length(x)) for _ in subjects]
    @threads for i in eachindex(subjects); vals[i],grads[i]=almquist_sensitivity_ode_subject_value_grad(subjects[i],x,etas[i],representation;dt=dt); end
    total=sum(vals); grad=vec(sum(reduce(hcat,grads),dims=2)); maybe_update_eta_cache!(eta_cache,subjects,etas,total,grad)
    total,grad,maxs,nc
end
function sensitivity_ode_mode_value(subjects,x,representation;dt=.25,maxiter_eta=30)
    etas,maxs,nc=warf_solve_all(subjects,x,representation;dt=dt,maxiter=maxiter_eta)
    vals=zeros(length(subjects))
    @threads for i in eachindex(subjects)
        h,_,_,G,_,_=warf_components(subjects[i],x,etas[i];dt=dt); F=cholesky(Symmetric((G+transpose(G))/2);check=true); vals[i]=2*(h+sum(log,diag(F.L)))
    end
    sum(vals),maxs,nc,etas
end