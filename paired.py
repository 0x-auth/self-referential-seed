"""Organism with a too-high / too-low pair (Heron bracket, vector form).
W predicts the next state x'.  V predicts its INVERSE PARTNER y' = x'/|x'|^2
(inversion through the unit sphere: the vector version of N/g).
Partner guess p2 = inv(V y).  The law uses the midpoint (p + p2)/2.
If x -> 0, its partner y -> infinity, so collapse can no longer read as zero error."""
import sys, numpy as np
sys.path.insert(0, ".")  # run from the self-referential-seed repo root
from duo_order import make, mv, DIM, SEEDS, STEPS, MU, ALIVE
from alive import classify, VAR_FLOOR
EPS = 1e-12
def inv(v): return v / (np.sum(v*v, axis=1, keepdims=True) + EPS)

def run(R, W, x, c, paired, steps=STEPS, tail=8000, seed=1):
    S,n = x.shape; W=W.copy(); x=x.copy()
    V = np.random.default_rng(seed).normal(0,0.3,(S,n,n))
    eta=np.full((S,n),0.05); etv=np.full((S,n),0.05); pv=np.ones(S)
    moved=np.zeros(S); scale=np.zeros(S); cnt=0; half=steps//2
    M2=np.zeros((S,n,n)); mean=np.zeros((S,n)); tc=0
    errx=np.zeros(S); width=np.zeros(S); dead=np.zeros(S,bool)
    for k in range(steps):
        p = mv(W,x)
        if paired:
            y = inv(x); q = mv(V,y); p2 = inv(q)
            pred = (p+p2)/2
        else:
            pred = p
        xn = np.tanh(mv(R,x) + c*pred)
        d = xn - p
        W = W + eta[:,:,None]*d[:,:,None]*x[:,None,:]
        m = np.linalg.norm(xn - pred, axis=1)
        if paired:
            yn = inv(xn); dy = yn - q
            sc = 1.0/(1.0 + np.sum(y*y,axis=1))[:,None,None]   # keep the inverse side's step bounded
            V = V + sc*etv[:,:,None]*dy[:,:,None]*y[:,None,:]
            m = m + np.minimum(np.linalg.norm(dy,axis=1), 1e6)  # collapse makes this side explode
            etv = np.clip(etv*(1+MU*np.sign(pv-m)[:,None]),1e-6,0.5)
        eta = np.clip(eta*(1+MU*np.sign(pv-m)[:,None]),1e-6,0.5); pv=m
        bad = ~np.isfinite(W).all(axis=(1,2)) | ~np.isfinite(V).all(axis=(1,2)) | ~np.isfinite(xn).all(axis=1)
        if bad.any():
            dead |= bad; W[bad]=0; V[bad]=0; xn[bad]=0
        if k>=half:
            moved+=np.linalg.norm(xn-x,axis=1); scale=np.maximum(scale,np.linalg.norm(xn,axis=1)); cnt+=1
        if k>=steps-tail:
            M2+=x[:,:,None]*x[:,None,:]; mean+=x; tc+=1
            errx+=np.linalg.norm(xn-p,axis=1)
            if paired: width+=np.linalg.norm(p-p2,axis=1)
        x=xn
    order=np.where(scale>VAR_FLOOR, moved/(cnt*np.maximum(scale,VAR_FLOOR)),0.0)
    mean/=tc; cov=M2/tc-mean[:,:,None]*mean[:,None,:]
    er=np.zeros(S); tr=np.zeros(S)
    for i,cv in enumerate(cov):
        lam=np.abs(np.linalg.eigvalsh((cv+cv.T)/2)); s1,s2=lam.sum(),(lam**2).sum()
        er[i]=s1**2/s2 if s2>1e-300 else 1.0; tr[i]=np.trace(cv)
    return order,er,tr,np.linalg.norm(x,axis=1),errx/tc,width/tc,dead

for c in (0.6, 1.0):
    R,W,_,x = make(0,c)
    print(f"\nc={c}, {SEEDS} seeds, {STEPS} steps")
    print(f"{'organism':<12}{'settled':>9}{'cycling':>9}{'explore':>9}{'collapsed |x|<1e-6':>21}{'diverged':>10}{'med err':>10}{'med width':>11}")
    for paired in (False, True):
        o,er,tr,xn,e,w,dead = run(R,W,x,c,paired)
        f,cy,ex = classify(o,er,tr)
        col = (xn<1e-6)&~dead
        name = "paired" if paired else "original"
        print(f"{name:<12}{f.mean()*100:>8.1f}%{cy.mean()*100:>8.1f}%{ex.mean()*100:>8.1f}%{col.mean()*100:>20.1f}%{dead.mean()*100:>9.1f}%{np.median(e[~dead]):>10.4f}{(np.median(w[~dead]) if paired else float('nan')):>11.4f}", flush=True)
