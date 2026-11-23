"""Weighted graph-projection splitting for convex composite regression.

This implementation is a sequential simulation of a distributed algorithm.
It never interprets elapsed time as distributed wall time. Data-factor maps and
structured-penalty maps use exactly the same graph projection code.

Problem: min_x sum_a f_a(sum_j A[a,j] x[j]) + sum_j r_j(x[j]),
where r_j is an elastic-net penalty. All weights and rho remain fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, xlogy


def soft(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@dataclass
class Factor:
    kind: str
    maps: dict[int, np.ndarray]
    response: np.ndarray | None = None
    scale: float = 1.0
    parameter: float = 1.0
    label: str = ""

    @property
    def size(self):
        if not self.maps:
            label = self.label or self.kind
            raise ValueError(
                f"Factor {label!r} has no incident map. Every retained factor "
                "must have an incident edge. For an all-zero data block, use "
                "make_regression(..., include_zero_edges=True) to retain its "
                "constant objective contribution."
            )
        return next(iter(self.maps.values())).shape[0]

    def value(self, z):
        if self.kind == "squared":
            return 0.5 * self.scale * np.sum((z - self.response)**2)
        if self.kind == "huber":
            r = np.abs(z - self.response)
            d = self.parameter
            return self.scale * np.sum(np.where(r <= d, 0.5*r*r, d*r-0.5*d*d))
        if self.kind == "quantile":
            r = self.response-z
            return self.scale * np.sum(np.maximum(self.parameter*r, (self.parameter-1)*r))
        if self.kind == "logistic":
            return self.scale * np.logaddexp(0, -self.response*z).sum()
        if self.kind == "l1":
            return self.scale * np.abs(z).sum()
        if self.kind == "l2":
            return self.scale * np.linalg.norm(z)
        raise ValueError(self.kind)

    def prox(self, v, coefficient):
        """argmin_z f(z)+coefficient/2 ||z-v||^2."""
        t = self.scale/coefficient
        if self.kind == "squared":
            return (v+t*self.response)/(1+t)
        if self.kind == "huber":
            a = v-self.response
            d = self.parameter
            return self.response + np.where(np.abs(a) <= d*(1+t), a/(1+t), a-t*d*np.sign(a))
        if self.kind == "quantile":
            a = v-self.response
            return self.response + np.where(a > t*(1-self.parameter), a-t*(1-self.parameter),
                                            np.where(a < -t*self.parameter, a+t*self.parameter, 0.0))
        if self.kind == "logistic":
            # The stationarity root lies between v and v+t*response.
            # Safeguard Newton: an unrestricted step can oscillate when rho is
            # small. A bracket fallback and explicit failure avoid silently
            # returning a nonproximal iterate.
            shift = t*self.response
            lo = v+np.minimum(0.0,shift)
            hi = v+np.maximum(0.0,shift)
            z = v.copy()
            for _ in range(100):
                q = expit(-self.response*z)
                residual = z-v-shift*q
                step = residual/(1+t*self.response**2*q*(1-q))
                tolerance = 2e-14*np.maximum(1.0,np.maximum(np.abs(v),np.abs(z)))
                converged = (np.abs(residual) <= tolerance) | (hi-lo <= tolerance)
                candidate = z-step
                if np.all(converged):
                    return np.minimum(hi,np.maximum(lo,candidate))
                lo = np.where(residual < 0,z,lo)
                hi = np.where(residual > 0,z,hi)
                inside = np.isfinite(candidate) & (candidate > lo) & (candidate < hi)
                z = np.where(converged,z,np.where(inside,candidate,lo+0.5*(hi-lo)))
            raise ArithmeticError("logistic proximal solve did not converge")
        if self.kind == "l1":
            return soft(v, t)
        if self.kind == "l2":
            return v * max(0.0, 1-t/max(np.linalg.norm(v), np.finfo(float).tiny))
        raise ValueError(self.kind)

    def project_dual(self, s):
        """Project onto dom f*; used only for a valid lower-bound certificate."""
        if self.kind == "squared":
            return s.copy()
        if self.kind == "huber":
            return np.clip(s, -self.scale*self.parameter, self.scale*self.parameter)
        if self.kind == "quantile":
            return np.clip(s, -self.scale*self.parameter, self.scale*(1-self.parameter))
        if self.kind == "logistic":
            u = np.clip(-s/(self.scale*self.response), 0, 1)
            return -self.scale*self.response*u
        if self.kind == "l1":
            return np.clip(s, -self.scale, self.scale)
        if self.kind == "l2":
            return s * min(1.0, self.scale/max(np.linalg.norm(s), np.finfo(float).tiny))
        raise ValueError(self.kind)

    def conjugate(self, s):
        if self.kind in ("squared", "huber"):
            return float(np.dot(s, self.response)+0.5*np.dot(s, s)/self.scale)
        if self.kind == "quantile":
            return float(np.dot(s, self.response))
        if self.kind == "logistic":
            u = np.clip(-s/(self.scale*self.response), 0, 1)
            return float(self.scale * np.sum(xlogy(u, u)+xlogy(1-u, 1-u)))
        if self.kind in ("l1", "l2"):
            return 0.0
        raise ValueError(self.kind)


@dataclass
class Problem:
    block_sizes: list[int]
    factors: list[Factor]
    l1: float = 0.0
    ridge: float = 0.0
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def objective(self, blocks):
        val = sum(f.value(sum(A@blocks[j] for j, A in f.maps.items())) for f in self.factors)
        return float(val+self.l1*sum(np.abs(b).sum() for b in blocks)
                     +0.5*self.ridge*sum(np.dot(b,b) for b in blocks))

    def dual_bound(self, multipliers):
        s = [f.project_dual(q) for f, q in zip(self.factors, multipliers)]
        at = [np.zeros(p) for p in self.block_sizes]
        for f, q in zip(self.factors, s):
            for j, A in f.maps.items():
                at[j] += A.T@q
        if self.ridge == 0:
            mx = max(np.max(np.abs(v)) for v in at)
            alpha = 1.0 if mx == 0.0 else min(1.0, self.l1/mx)
            s = [alpha*q for q in s]
            at = [alpha*v for v in at]
            rc = 0.0
        else:
            rc = sum(np.dot(soft(v,self.l1),soft(v,self.l1)) for v in at)/(2*self.ridge)
        return float(-sum(f.conjugate(q) for f,q in zip(self.factors,s))-rc)


class GraphSplitting:
    """Full product-space implementation; all vector slots are retained for audit.

    Communication counts assume one uplink and one downlink of each graph
    output and coefficient vector per iteration: 2 sum_edges(m_a+p_j).
    Certificates require a separate gather and are not counted as optimization
    traffic; the experiment scripts state their evaluation frequency.
    """
    def __init__(self, problem, rho=1.0, weights="unit", custom_weights=None):
        self.problem=problem
        self.rho=float(rho)
        if self.rho <= 0: raise ValueError("rho must be positive")
        factor_sizes=[f.size for f in problem.factors]
        self.edges=[(a,j,A) for a,f in enumerate(problem.factors) for j,A in f.maps.items()]
        na,nj=len(problem.factors),len(problem.block_sizes)
        da=np.bincount([a for a,j,A in self.edges],minlength=na)
        dj=np.bincount([j for a,j,A in self.edges],minlength=nj)
        self.da,self.dj=da,dj
        # a,c,b,d denote factor anchor, feature anchor, edge output, edge copy.
        if custom_weights is not None:
            self.a,self.c,self.b,self.d=[np.asarray(custom_weights[k],dtype=float) for k in ("a","c","b","d")]
        elif weights == "unit":
            self.a=np.ones(na);self.c=np.ones(nj)
            self.b=np.ones(len(self.edges));self.d=np.ones(len(self.edges))
        elif weights == "copy_normalized":
            self.a=np.ones(na);self.c=np.full(nj,0.5)
            self.b=np.ones(len(self.edges));self.d=np.array([0.5/dj[j] for a,j,A in self.edges])
        elif weights in ("degree_balanced", "budget"):
            self.a=np.ones(na);self.c=np.ones(nj)
            self.b=np.array([da[a] for a,j,A in self.edges],dtype=float)
            self.d=np.array([1.0/dj[j] for a,j,A in self.edges])
        else: raise ValueError(weights)
        self.weights=weights
        if any(np.any(v<=0) for v in (self.a,self.c,self.b,self.d)): raise ValueError("all weights must be positive")
        self.ys=[];self.xs=[];self.yes=[];self.xes=[];nw=[];off=0
        for n,w in zip(factor_sizes,self.a):
            self.ys.append(slice(off,off+n));off+=n;nw.extend([w]*n)
        for n,w in zip(problem.block_sizes,self.c):
            self.xs.append(slice(off,off+n));off+=n;nw.extend([w]*n)
        for e,(a,j,A) in enumerate(self.edges):
            n=A.shape[0];self.yes.append(slice(off,off+n));off+=n;nw.extend([self.b[e]]*n)
            n=A.shape[1];self.xes.append(slice(off,off+n));off+=n;nw.extend([self.d[e]]*n)
        self.W=np.asarray(nw)
        self.eta=np.zeros(off);self.u=np.zeros(off)
        self.edge_by_factor=[[e for e,(aa,j,A) in enumerate(self.edges) if aa==a] for a in range(na)]
        self.edge_by_feature=[[e for e,(a,jj,A) in enumerate(self.edges) if jj==j] for j in range(nj)]
        self.chol=[]
        self.solve_side=[]
        for e,(a,j,A) in enumerate(self.edges):
            # Use the smaller of coefficient-space and observation-space solves.
            if A.shape[1] <= A.shape[0]:
                self.chol.append(cho_factor(self.d[e]*np.eye(A.shape[1])+self.b[e]*(A.T@A),check_finite=False))
                self.solve_side.append("coefficient")
            else:
                self.chol.append(cho_factor(self.d[e]*np.eye(A.shape[0])+self.b[e]*(A@A.T),check_finite=False))
                self.solve_side.append("observation")
        self.communication_per_iteration=2*sum(A.shape[0]+A.shape[1] for a,j,A in self.edges)
        self.factorization_storage_scalars=sum(min(A.shape)**2 for a,j,A in self.edges)
        self.largest_local_matrix=max(min(A.shape) for a,j,A in self.edges)

    def prox(self,v):
        out=np.empty_like(v)
        for a,(f,s) in enumerate(zip(self.problem.factors,self.ys)):
            out[s]=f.prox(v[s],self.rho*self.a[a])
        for j,s in enumerate(self.xs):
            coeff=self.rho*self.c[j]
            out[s]=soft(v[s],self.problem.l1/coeff)/(1+self.problem.ridge/coeff)
        for e,(a,j,A) in enumerate(self.edges):
            sy,sx=self.yes[e],self.xes[e]
            if self.solve_side[e]=="coefficient":
                x=cho_solve(self.chol[e],self.d[e]*v[sx]+self.b[e]*(A.T@v[sy]),check_finite=False)
            else:
                x=v[sx]+self.b[e]*A.T@cho_solve(self.chol[e],v[sy]-A@v[sx],check_finite=False)
            out[sx]=x;out[sy]=A@x
        return out

    def project(self,v):
        out=v.copy()
        for j,s in enumerate(self.xs):
            es=self.edge_by_feature[j]
            mean=(self.c[j]*v[s]+sum(self.d[e]*v[self.xes[e]] for e in es))/(self.c[j]+sum(self.d[e] for e in es))
            out[s]=mean
            for e in es:out[self.xes[e]]=mean
        for a,s in enumerate(self.ys):
            es=self.edge_by_factor[a]
            residual=v[s]-sum(v[self.yes[e]] for e in es)
            delta=residual/(1/self.a[a]+sum(1/self.b[e] for e in es))
            out[s]=v[s]-delta/self.a[a]
            for e in es:out[self.yes[e]]=v[self.yes[e]]+delta/self.b[e]
        return out

    def certificate(self):
        blocks=[self.eta[s].copy() for s in self.xs]
        multipliers=[-self.rho*self.a[a]*self.u[s] for a,s in enumerate(self.ys)]
        primal=self.problem.objective(blocks)
        dual=self.problem.dual_bound(multipliers)
        if dual-primal > 1e-9*max(1.0,abs(primal)):
            raise ArithmeticError("Fenchel lower bound exceeds the feasible objective")
        return primal,dual,max(primal-dual,0.0)

    def invariant_error(self):
        primal=np.linalg.norm(self.eta-self.project(self.eta))
        dual=np.linalg.norm(self.project(self.u))
        return float(primal),float(dual)

    def solve(self,tol=1e-6,maxiter=20000,check_every=10,snapshot_tolerances=()):
        """Return the fitted coefficients, certificates and iteration diagnostics.

        ``state_dimension`` is the scalar dimension of ONE full lifted vector
        (``eta``): sum_a m_a + sum_j p_j + sum_edges(m_a+p_j). It does not
        measure the total allocated memory: ``u``, iteration temporaries,
        local maps and cached factorizations are separate arrays.
        """
        start=perf_counter();history=[];snapshots={};laststep=np.nan
        initial_objective=self.problem.objective([np.zeros(p) for p in self.problem.block_sizes])
        for k in range(1,maxiter+1):
            xi=self.prox(self.eta-self.u)
            neweta=self.project(xi)  # u is W-orthogonal to the consistency space.
            du=xi-neweta;deta=neweta-self.eta
            laststep=float(self.rho*np.dot(self.W,deta*deta+du*du))
            self.u+=du;self.eta=neweta
            if k%check_every==0 or k==1 or k==maxiter:
                primal,dual,gap=self.certificate()
                relative=gap/max(1.0,abs(primal))
                history.append(dict(iteration=k,objective=primal,dual=dual,gap=gap,relative_gap=relative,
                                    squared_step=laststep,elapsed_seconds=perf_counter()-start,
                                    communicated_scalars=k*self.communication_per_iteration))
                for threshold in snapshot_tolerances:
                    if threshold not in snapshots and relative<=threshold:
                        snapshots[threshold]=dict(iteration=k,gap=gap,relative_gap=relative,
                                                  coef=np.concatenate([self.eta[s] for s in self.xs]))
                if relative <= tol:break
        elapsed=perf_counter()-start
        primal,dual,gap=self.certificate()
        return dict(coef=np.concatenate([self.eta[s] for s in self.xs]),iterations=k,objective=primal,
                    dual=dual,gap=gap,relative_gap=gap/max(1.0,abs(primal)),
                    squared_step=laststep,elapsed_seconds=elapsed,converged=(gap/max(1.0,abs(primal))<=tol),
                    communicated_scalars=k*self.communication_per_iteration,
                    communication_per_iteration=self.communication_per_iteration,
                    largest_local_matrix=self.largest_local_matrix,
                    factorization_storage_scalars=self.factorization_storage_scalars,
                    state_dimension=self.eta.size,invariant_error=self.invariant_error(),history=history,
                    diagnostic_evaluations=len(history)+1,snapshots=snapshots,
                    initial_objective=initial_objective)


def make_regression(X,y,row_parts=4,column_parts=4,l1=0.02,ridge=0.01,
                    loss="squared",loss_parameter=1.0,fusion=0.0,fusion_parts=None,
                    scale_design=True,include_zero_edges=False,name=""):
    """Build a regression graph, preserving nonzero map incidences.

    Logistic labels may be encoded as 0/1 or -1/+1.
    Designs are divided by sqrt(n), an exact change of variables. For Huber,
    the threshold and response are divided by sqrt(n). Quantile has loss scale
    1/sqrt(n) after this transformation. Logistic factors use labels multiplied
    by sqrt(n) and loss scale 1/n. Every objective equals mean(loss)+penalties.
    Fusion factors group consecutive rows of D.
    Every retained factor must have an incident edge. If a whole data row
    block is zero, set include_zero_edges=True to keep its constant loss;
    otherwise solver construction raises a descriptive ValueError.
    """
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float);n,p=X.shape
    if loss == "logistic":
        if np.all(np.isin(y, (0, 1))):
            y = 2*y - 1
        elif not np.all(np.isin(y, (-1, 1))):
            raise ValueError("logistic responses must use either 0/1 or -1/+1 labels")
    rows=np.array_split(np.arange(n),row_parts)
    cols=np.array_split(np.arange(p),column_parts)
    if any(len(c)==0 for c in cols) or any(len(r)==0 for r in rows):raise ValueError("empty partition")
    factors=[]
    use_scale=scale_design
    divisor=np.sqrt(n) if use_scale else 1.0
    fscale=(1.0 if loss in ("squared","huber") else (1/np.sqrt(n) if loss=="quantile" else 1/n)) if use_scale else 1/n
    param=loss_parameter/divisor if loss=="huber" else loss_parameter
    for i,rr in enumerate(rows):
        maps={j:np.ascontiguousarray(X[np.ix_(rr,cc)]/divisor) for j,cc in enumerate(cols)}
        if not include_zero_edges:maps={j:A for j,A in maps.items() if np.any(A)}
        response=y[rr]*divisor if loss=="logistic" else y[rr]/divisor
        factors.append(Factor(loss,maps,response,scale=fscale,parameter=param,label=f"data_{i}"))
    if fusion>0:
        D=np.diff(np.eye(p),axis=0)
        fparts=column_parts if fusion_parts is None else fusion_parts
        for i,rr in enumerate(np.array_split(np.arange(p-1),fparts)):
            if not len(rr):continue
            maps={j:np.ascontiguousarray(D[np.ix_(rr,cc)]) for j,cc in enumerate(cols)}
            if not include_zero_edges:maps={j:A for j,A in maps.items() if np.any(A)}
            factors.append(Factor("l1",maps,scale=fusion,label=f"difference_{i}"))
    return Problem([len(c) for c in cols],factors,l1,ridge,name,
                   dict(n=n,p=p,row_parts=row_parts,column_parts=column_parts,loss=loss,
                        loss_parameter=loss_parameter,fusion=fusion,design_divisor=divisor,
                        include_zero_edges=include_zero_edges))
