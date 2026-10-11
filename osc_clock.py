"""Organism with the E1/E2 clock: the self-prediction enters its own law with
weight c*s_n, where s_n alternates t (E1, forward) and -1/t (E2, time-mirrored).
Same setup and fate classifier as alive.py (200 seeds, 40k steps, dim 6)."""
import sys, numpy as np
sys.path.insert(0, ".")  # run from the self-referential-seed repo root
from duo_order import make, mv, DIM, SEEDS, STEPS, MU, ALIVE
from alive import classify, VAR_FLOOR

def run(R, W, x, c, sched, steps=STEPS, tail=8000):
    S,n = x.shape; W=W.copy(); x=x.copy()
    eta=np.full((S,n),0.05); pv=np.ones(S)
    moved=np.zeros(S); scale=np.zeros(S); cnt=0; half=steps//2
    M2=np.zeros((S,n,n)); mean=np.zeros((S,n)); tc=0; err_tail=np.zeros(S)
    for k in range(steps):
        s = sched(k)
        p = mv(W,x)
        xn = np.tanh(mv(R,x) + c*s*p)
        d = xn - p; m = np.linalg.norm(d,axis=1)
        W = W + eta[:,:,None]*d[:,:,None]*x[:,None,:]
        eta = np.clip(eta*(1+MU*np.sign(pv-m)[:,None]),1e-6,0.5); pv=m
        if not np.all(np.isfinite(W)): W=np.nan_to_num(W)
        if k>=half:
            moved+=np.linalg.norm(xn-x,axis=1); scale=np.maximum(scale,np.linalg.norm(xn,axis=1)); cnt+=1
        if k>=steps-tail:
            M2+=x[:,:,None]*x[:,None,:]; mean+=x; tc+=1; err_tail+=m
        x=xn
    order=np.where(scale>VAR_FLOOR, moved/(cnt*np.maximum(scale,VAR_FLOOR)),0.0)
    mean/=tc; cov=M2/tc-mean[:,:,None]*mean[:,None,:]
    er=np.zeros(S); tr=np.zeros(S)
    for i,cv in enumerate(cov):
        lam=np.abs(np.linalg.eigvalsh((cv+cv.T)/2)); s1,s2=lam.sum(),(lam**2).sum()
        er[i]=s1**2/s2 if s2>1e-300 else 1.0; tr[i]=np.trace(cv)
    return order,er,tr,err_tail/tc

PHI=(1+5**.5)/2
rng_sign=np.random.default_rng(99)
RANDSEQ=rng_sign.integers(0,2,STEPS)
def alt(t):  return lambda k: t if k%2==0 else -1/t
def rnd(t):  return lambda k: t if RANDSEQ[k]==0 else -1/t   # same values, random order
def const(v):return lambda k: v

c=0.6
R,W,_,x = make(0,c)
conds = [
 ("baseline  s=1 (original)",        const(1.0)),
 ("const s=+0.5 (half coupling)",    const(0.5)),
 ("const s=0   (no self-ref)",       const(0.0)),
 ("ALT t=1     (+1,-1)",             alt(1.0)),
 ("ALT t=phi   (+1.618,-0.618)",     alt(PHI)),
 ("ALT t=2     (+2,-0.5)",           alt(2.0)),
 ("ALT t=0.5   (+0.5,-2)",           alt(0.5)),
 ("ALT t=1/phi (+0.618,-1.618)",     alt(1/PHI)),
 ("RANDOM order, t=1",               rnd(1.0)),
 ("RANDOM order, t=phi",             rnd(PHI)),
]
print(f"c={c}, {SEEDS} seeds, {STEPS} steps, dim {DIM}\n")
print(f"{'condition':<32}{'settled':>9}{'cycling':>9}{'explore':>9}{'med err(alive)':>16}")
for name,sch in conds:
    o,er,tr,e = run(R,W,x,c,sch)
    f,cy,ex = classify(o,er,tr); a=~f
    print(f"{name:<32}{f.mean()*100:>8.1f}%{cy.mean()*100:>8.1f}%{ex.mean()*100:>8.1f}%{np.median(e[a]) if a.sum() else float('nan'):>16.4f}", flush=True)
