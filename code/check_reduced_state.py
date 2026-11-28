"""Compare the compressed and full weighted graph ADMM states (NumPy only)."""

import numpy as np


def block_slices(dimensions, offset):
    blocks = {}
    for key, size in dimensions.items():
        blocks[key] = slice(offset, offset + size)
        offset += size
    return blocks, offset


def build_case(rng):
    output_sizes = {0: 2, 1: 3, 2: 1}
    coefficient_sizes = {0: 3, 1: 2}
    edges = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
    maps = {e: rng.normal(size=(output_sizes[e[0]], coefficient_sizes[e[1]]))
            for e in edges}
    maps[0, 0][1] = maps[0, 0][0]  # A rank-deficient block.
    h, offset = block_slices(output_sizes, 0)
    beta, offset = block_slices(coefficient_sizes, offset)
    he, offset = block_slices({e: output_sizes[e[0]] for e in edges}, offset)
    be, size = block_slices({e: coefficient_sizes[e[1]] for e in edges}, offset)

    # C x = 0 imposes consistency, and G x = 0 imposes local edge graphs.
    constraints = []
    graphs = []
    for a in h:
        rows = np.zeros((output_sizes[a], size))
        rows[:, h[a]] = np.eye(output_sizes[a])
        for e in edges:
            if e[0] == a:
                rows[:, he[e]] = -np.eye(output_sizes[a])
        constraints.append(rows)
    for e in edges:
        a, j = e
        rows = np.zeros((coefficient_sizes[j], size))
        rows[:, be[e]] = np.eye(coefficient_sizes[j])
        rows[:, beta[j]] = -np.eye(coefficient_sizes[j])
        constraints.append(rows)
        rows = np.zeros((output_sizes[a], size))
        rows[:, he[e]] = np.eye(output_sizes[a])
        rows[:, be[e]] = -maps[e]
        graphs.append(rows)
    return (h, beta, he, be), maps, np.vstack(constraints), np.vstack(graphs)


def minimal_state(eta, dual, blocks):
    h, beta, he, be = blocks
    return tuple({key: x[s].copy() for key, s in family.items()}
                 for x, family in ((eta, beta), (eta, he), (dual, h), (dual, be)))


def reconstruct(state, blocks, weights):
    h, beta, he, be = blocks
    eta_beta, eta_he, u_h, u_be = state
    eta = np.zeros_like(weights)
    dual = np.zeros_like(weights)
    for a in h:
        dual[h[a]] = u_h[a]
    for j in beta:
        eta[beta[j]] = eta_beta[j]
    for e in he:
        a, j = e
        eta[he[e]], eta[be[e]] = eta_he[e], eta_beta[j]
        eta[h[a]] += eta_he[e]
        dual[be[e]] = u_be[e]
        dual[he[e]] = -weights[h[a]][0] / weights[he[e]][0] * u_h[a]
        dual[beta[j]] -= weights[be[e]][0] / weights[beta[j]][0] * u_be[e]
    return eta, dual


def step_compressed(state, blocks, maps, weights, rho, curvature, linear):
    h, beta, he, be = blocks
    eta_beta, eta_he, u_h, u_be = state
    eta_h = {a: np.zeros_like(u_h[a]) for a in h}
    u_beta = {j: np.zeros_like(eta_beta[j]) for j in beta}
    for e in maps:
        a, j = e
        eta_h[a] += eta_he[e]
        u_beta[j] -= weights[be[e]][0] / weights[beta[j]][0] * u_be[e]

    def anchor(s, q):
        return (rho * weights[s] * q + linear[s]) / (curvature[s] + rho * weights[s])

    xh = {a: anchor(s, eta_h[a] - u_h[a]) for a, s in h.items()}
    xb = {j: anchor(s, eta_beta[j] - u_beta[j]) for j, s in beta.items()}
    xb_edge, xh_edge = {}, {}
    for e, A in maps.items():
        a, j = e
        b, d = weights[he[e]][0], weights[be[e]][0]
        qh = eta_he[e] + weights[h[a]][0] / b * u_h[a]
        qb = eta_beta[j] - u_be[e]
        rhs = d * qb + b * A.T @ qh
        xb_edge[e] = np.linalg.solve(d * np.eye(A.shape[1]) + b * A.T @ A, rhs)
        xh_edge[e] = A @ xb_edge[e]

    new_he, new_u_h = {}, {}
    for a, anchor_slice in h.items():
        incident = [e for e in maps if e[0] == a]
        w = weights[anchor_slice][0]
        denominator = 1 / w + sum(1 / weights[he[e]][0] for e in incident)
        correction = (xh[a] - sum(xh_edge[e] for e in incident)) / denominator
        new_u_h[a] = u_h[a] + correction / w
        for e in incident:
            new_he[e] = xh_edge[e] + correction / weights[he[e]][0]
    new_beta, new_u_be = {}, {}
    for j, anchor_slice in beta.items():
        incident = [e for e in maps if e[1] == j]
        w = weights[anchor_slice][0]
        denominator = w + sum(weights[be[e]][0] for e in incident)
        new_beta[j] = (w * xb[j] + sum(weights[be[e]][0] * xb_edge[e]
                                       for e in incident)) / denominator
        for e in incident:
            new_u_be[e] = u_be[e] + xb_edge[e] - new_beta[j]
    return new_beta, new_he, new_u_h, new_u_be


def check(seed, zero_start):
    rng = np.random.default_rng(seed)
    blocks, maps, C, G = build_case(rng)
    h, beta, he, be = blocks
    size = C.shape[1]
    weights = np.zeros(size)
    curvature = np.zeros(size)
    linear = np.zeros(size)
    rho = 1.7
    for family in blocks:
        for s in family.values():
            weights[s] = rng.uniform(0.35, 2.6)
    for s in [*h.values(), *beta.values()]:
        curvature[s] = rng.uniform(0.4, 1.5, size=s.stop - s.start)
        linear[s] = rng.normal(size=s.stop - s.start)

    # The reference solves one dense KKT system for all quadratic proximals
    # and graph constraints, followed by a generic weighted null-space projection.
    W_inv_CT = C.T / weights[:, None]
    projection = np.eye(size) - W_inv_CT @ np.linalg.solve(C @ W_inv_CT, C)
    proximal = np.block([[np.diag(curvature + rho * weights), G.T],
                         [G, np.zeros((G.shape[0], G.shape[0]))]])
    eta = np.zeros(size) if zero_start else projection @ rng.normal(size=size)
    dual = np.zeros(size) if zero_start else W_inv_CT @ rng.normal(size=C.shape[0])
    state = minimal_state(eta, dual, blocks)
    if sum(x.size for family in state for x in family.values()) != size:
        raise AssertionError("Unexpected minimal-state size")
    err = 0.0
    for k in range(41):
        eta_red, dual_red = reconstruct(state, blocks, weights)
        delta = np.r_[eta_red - eta, dual_red - dual]
        if not np.isfinite(delta).all():
            raise AssertionError("Nonfinite state in reconstruction test")
        err = max(err, np.max(np.abs(delta)))
        if k == 40:
            break
        rhs = np.r_[rho * weights * (eta - dual) + linear, np.zeros(G.shape[0])]
        xi = np.linalg.solve(proximal, rhs)[:size]
        new_eta = projection @ (xi + dual)
        dual = dual + xi - new_eta
        eta = new_eta
        state = step_compressed(state, blocks, maps, weights, rho, curvature, linear)
    if not np.isfinite(err) or err > 2e-12:
        raise AssertionError(f"Full and reduced states differ by {err:.3e}")
    return err


if __name__ == "__main__":
    errors = [check(1701, True), check(1702, False)]
    print(f"PASS: 40 full/reduced updates per start, max absolute error {max(errors):.3e}")
