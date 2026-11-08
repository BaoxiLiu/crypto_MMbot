"""Finite inventory-grid successor features from paper Appendix C.

Inventory is measured in fixed order units, rewards in tick times reference size.
The market is frozen during a solve. Execution remains the common OMS's task.
"""
import numpy as np
from numba import njit


@njit
def solve_hjb(grid, actions, rates, costs, objective, steps, dt, discount, risk_scale):
    count = len(grid)
    previous = np.zeros((count, 4))
    choices = np.zeros(count, np.int64)
    probabilities = np.zeros((count, len(actions), 2))
    immediate = np.zeros((count, len(actions), 4))
    for j in range(count):
        q = grid[j]
        for k in range(len(actions)):
            bid, ask = actions[k]
            pb = 0. if bid < 0 or j == count-1 else -np.expm1(-rates[0]*np.exp(-rates[2]*bid)*dt)
            pa = 0. if ask < 0 or j == 0 else -np.expm1(-rates[1]*np.exp(-rates[3]*ask)*dt)
            probabilities[j,k,0] = pb
            probabilities[j,k,1] = pa
            immediate[j,k,0] = pb*max(0.,bid)+pa*max(0.,ask)
            immediate[j,k,1] = q+pb-pa
            immediate[j,k,2] = (q*q+2*q*(pb-pa)+pb+pa-2*pb*pa)*risk_scale
            immediate[j,k,3] = pb*costs[0]+pa*costs[1]
    for _ in range(steps):
        current = np.zeros((count, 4))
        for j in range(count):
            best = -np.inf
            for k in range(len(actions)):
                pb, pa = probabilities[j,k]
                stay = (1-pb)*(1-pa)+pb*pa
                up = pb*(1-pa)
                down = (1-pb)*pa
                score = 0.
                for coordinate in range(4):
                    continuation = (stay*previous[j,coordinate]+
                                    up*previous[min(count-1,j+1),coordinate]+
                                    down*previous[max(0,j-1),coordinate])
                    score += objective[coordinate]*(immediate[j,k,coordinate]+discount*continuation)
                if score > best:
                    best = score
                    choices[j] = k
                    for coordinate in range(4):
                        continuation = (stay*previous[j,coordinate]+
                                        up*previous[min(count-1,j+1),coordinate]+
                                        down*previous[max(0,j-1),coordinate])
                        current[j,coordinate] = immediate[j,k,coordinate]+discount*continuation
        previous = current
    return choices, previous


def fit_objective(features, rewards, weights, prior, ridge, mixing, max_inventory):
    """Appendix C residual ridge, safe-prior mixture and inventory projection."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(rewards, dtype=float)-x[:, 0]
    x = x[:, 1:]
    w = np.asarray(weights, dtype=float)
    w = w/w.sum()
    beta = np.linalg.solve((x.T*w)@x+ridge*np.eye(3), x.T@(w*y))
    objective = np.asarray(prior, dtype=float).copy()
    objective[1:] = (1-mixing)*objective[1:]+mixing*beta
    objective[0] = 1.
    objective[2] = min(-1e-8, objective[2])
    objective[3] = min(0., objective[3])
    limit = -2*objective[2]*max_inventory
    objective[1] = np.clip(objective[1], -limit, limit)
    return objective
