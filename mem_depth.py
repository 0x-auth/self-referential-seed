"""Memory depth per fate. Question: given the present state x_t, how much do
OLDER states (x_{t-1}..x_{t-k}) add to predicting the next move?
Target is arctanh(x_{t+1}) = (R + cW_t) x_t, linear in x_t if the self-model
has stopped changing. If the past still matters, older states help.
gain(k) = 1 - err(lags 0..k) / err(lag 0 only), on a held-out later window."""
import sys, numpy as np
sys.path.insert(0, ".")  # run from the self-referential-seed repo root
from duo_order import make, mv, DIM, SEEDS, STEPS, MU
from alive import classify, VAR_FLOOR

def run(R, W, x, c, steps=STEPS, tail=8000):
    S,n=x.shape; W=W.copy(); x=x.copy(); eta=np.full((S,n),0.05); pv=np.ones(S)
    moved=np.zeros(S); scale=np.zeros(S); cnt=0; half=steps//2
    traj=np.zeros((tail+1,S,n)); Wstart=None
    for k in range(steps):
        p=mv(W,x); xn=np.tanh(mv(R,x)+c*p); d=xn-p; m=np.linalg.norm(d,axis=1)
        W=W+eta[:,:,None]*d[:,:,None]*x[:,None,:]
        eta=np.clip(eta*(1+MU*np.sign(pv-m)[:,None]),1e-6,0.5); pv=m
        if k>=half: moved+=np.linalg.norm(xn-x,axis=1); scale=np.maximum(scale,np.linalg.norm(xn,axis=1)); cnt+=1
        if k==steps-tail-1: Wstart=W.copy(); traj[0]=xn
        if k>=steps-tail: traj[k-(steps-tail)+1]=xn
        x=xn
    order=np.where(scale>VAR_FLOOR,moved/(cnt*np.maximum(scale,VAR_FLOOR)),0.0)
    X=traj[:-1]; cov=np.einsum('tsi,tsj->sij',X,X)/len(X)-np.einsum('si,sj->sij',X.mean(0),X.mean(0))
    er=np.zeros(S); tr=np.zeros(S)
    for i,cv in enumerate(cov):
        lam=np.abs(np.linalg.eigvalsh((cv+cv.T)/2)); s1,s2=lam.sum(),(lam**2).sum()
        er[i]=s1**2/s2 if s2>1e-300 else 1.0; tr[i]=np.trace(cv)
    wdrift=np.linalg.norm(W-Wstart,axis=(1,2))/np.maximum(np.linalg.norm(W,axis=(1,2)),1e-12)
    return order,er,tr,traj,wdrift

def gains(traj, K=4, ridge=1e-8):
    T,S,n=traj.shape; out=np.full((S,K+1),np.nan)
    Y=np.arctanh(np.clip(traj[K+1:],-0.999999,0.999999))
    for s in range(S):
        feats=[traj[K-j:T-1-j,s] for j in range(K+1)]  # lag 0..K
        y=Y[:,s]; split=len(y)//2; base=None
        for k in range(K+1):
            F=np.hstack(feats[:k+1]); Ftr,Fte,ytr,yte=F[:split],F[split:],y[:split],y[split:]
            A=Ftr.T@Ftr+ridge*np.eye(F.shape[1]); B=np.linalg.solve(A,Ftr.T@ytr)
            e=np.mean((Fte@B-yte)**2)
            if k==0: base=e
            out[s,k]=1-e/base if base>1e-20 else np.nan
    return out

if __name__ == "__main__":
    from scipy.stats import mannwhitneyu
    for c in (0.6, 1.0):
        R,W,_,x=make(0,c); o,er,tr,traj,wd=run(R,W,x,c); f,cy,ex=classify(o,er,tr)
        g=gains(traj)
        present_exact=np.isnan(g[:,4])      # x_t alone predicts the next move exactly
        g0=np.where(present_exact,0.0,g[:,4])
        print(f"\nc={c}, {SEEDS} seeds   (settled runs have no movement to predict; excluded)")
        print(f"{'fate':<11}{'n':>4}{'present alone enough':>22}{'past gain (k=4)':>17}{'self-model change':>19}")
        for name,mask in (("cycling",cy),("exploring",ex)):
            print(f"{name:<11}{mask.sum():>4}{present_exact[mask].mean()*100:>21.1f}%{np.median(g0[mask]):>17.3f}{np.median(wd[mask]):>19.2e}")
        print(f"  exploring vs cycling: past gain p={mannwhitneyu(g0[ex],g0[cy]).pvalue:.1e}, "
              f"self-model change p={mannwhitneyu(wd[ex],wd[cy]).pvalue:.1e}")
