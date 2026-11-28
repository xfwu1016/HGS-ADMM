"""One fixed-metric continuation selected from paired state increments.

The rule minimizes a surrogate of the suffix-distance numerator, not actual
stopping cost. It has no oracle-regret guarantee for general trajectories.
"""
import numpy as np
from pilot_budget_experiment import water_fill
from hybrid_solver import GraphSplitting

def select_metric(solver, previous_eta, previous_lambda, *, joint=True,
                  uniform=False, floor_fraction=.05, rho_bounds=(1/256,4)):
    delta_eta=solver.eta-previous_eta
    delta_lambda=solver.rho*solver.W*solver.u-previous_lambda
    ne=len(solver.edges)
    omega=np.zeros(ne);pi=np.zeros(ne)
    for indices in solver.edge_by_factor:
        values=np.ones(len(indices)) if uniform else np.array([np.linalg.norm(delta_eta[solver.yes[e]]) for e in indices])
        omega[indices]=water_fill(values,floor_fraction)
    for indices in solver.edge_by_feature:
        values=np.array([np.linalg.norm(delta_lambda[solver.xes[e]]) for e in indices]) if joint and not uniform else np.ones(len(indices))
        pi[indices]=water_fill(values,floor_fraction)
    weights=dict(a=np.ones_like(solver.a),c=np.ones_like(solver.c),b=1/omega,d=pi)
    # Assemble diagonal coordinates without constructing or refactoring a solver.
    W=np.empty_like(solver.W)
    for ix in solver.ys+solver.xs:W[ix]=1
    for e in range(ne):W[solver.yes[e]]=1/omega[e];W[solver.xes[e]]=pi[e]
    P=float(np.dot(W,delta_eta**2));D=float(np.dot(1/W,delta_lambda**2))
    # If one increment vanishes, do not extrapolate a degenerate balance ratio.
    rho=float(np.clip(np.sqrt(D/P),*rho_bounds)) if P>0 and D>0 else solver.rho
    return weights,rho,dict(P=P,D=D,omega=omega.tolist(),pi=pi.tolist(),rho_clipped=(rho in rho_bounds))

def apply_metric(solver,weights,rho):
    selected=GraphSplitting(solver.problem,rho=rho,custom_weights=weights)
    selected.eta[:]=solver.eta
    selected.u[:]=solver.rho*solver.W*solver.u/(rho*selected.W)
    np.testing.assert_allclose(rho*selected.W*selected.u,solver.rho*solver.W*solver.u,rtol=1e-13,atol=1e-13)
    if max(selected.invariant_error())>1e-8:raise ArithmeticError('State transfer failed')
    return selected
