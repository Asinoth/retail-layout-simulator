"""Analytical optimum for a SyntheticShop.

Works strictly off the closed-form revenue model in
synthetic_shops.analytical_revenue. The GA's simulator-backed fitness
(viz_ga._ga_compute_layout_score) is structurally independent -- agent
paths, heatmap, queue, abandonment -- so any ranking agreement between
the two is informative rather than circular.

The anti-circularity guarantee is the whole point: this module imports
neither viz_ga nor viz_optimize nor simulation; its only project
dependency is synthetic_shops. Enforced by test_oracle_anti_circularity().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from synthetic_shops import (
    SyntheticShop,
    SyntheticItem,
    analytical_revenue,
    grid_layout_within_sections,
)


@dataclass
class OracleResult:
    layout: Dict[str, Tuple[float, float]]
    true_revenue: float
    converged: bool
    n_iter: int
    method: str
    note: str = ""


def _layout_to_vec(layout: Dict[str, Tuple[float, float]],
                   ordered_names: List[str]) -> np.ndarray:
    """Flatten a layout dict to a (2*N,) vector in the order of ordered_names."""
    return np.array(
        [coord for name in ordered_names for coord in layout[name]],
        dtype=np.float64,
    )


def _vec_to_layout(vec: np.ndarray,
                   ordered_names: List[str]) -> Dict[str, Tuple[float, float]]:
    return {ordered_names[i]: (float(vec[2 * i]), float(vec[2 * i + 1]))
            for i in range(len(ordered_names))}


def _bounds_for(shop: SyntheticShop,
                ordered_names: List[str]) -> List[Tuple[float, float]]:
    """Per-coordinate bounds: each item must stay within its own section's
    rectangle (with a small pad). Returns a list of (lo, hi) tuples in the
    same flattened order as ``_layout_to_vec``.

    Falling back to global shop bounds for items whose section is missing
    (shouldn't happen for generated synthetic shops, but defensive)."""
    bounds: List[Tuple[float, float]] = []
    by_name = {it.name: it for it in shop.items}
    for name in ordered_names:
        it = by_name[name]
        w, h = it.size
        sec = shop.sections.get(it.category)
        if sec is None:
            bounds.append((0.1, max(0.1, shop.width - w - 0.1)))
            bounds.append((0.1, max(0.1, shop.height - h - 0.1)))
        else:
            sx, sy, sw, sh = sec
            pad = 0.05
            bounds.append((sx + pad, max(sx + pad, sx + sw - w - pad)))
            bounds.append((sy + pad, max(sy + pad, sy + sh - h - pad)))
    return bounds


def solve_oracle(shop: SyntheticShop,
                 n_restarts: int = 6,
                 method: str = "L-BFGS-B") -> OracleResult:
    """Solve max R(layout) over per-item-section bounds.

    Uses multi-start L-BFGS-B (analytic gradient via finite differences;
    the objective is smooth in positions). Initial points: the grid
    layout + (n_restarts - 1) random perturbations within bounds.

    Why L-BFGS-B: bounds-constrained, gradient-aware, handles ~20-D
    smoothly. SLSQP would be equivalent but L-BFGS-B has better
    convergence behavior on this objective family. We avoid global
    methods (basin-hopping, differential evolution) because the
    objective is smooth and the multi-start is sufficient.

    Non-overlap is NOT a constraint in the optimization: items are
    confined to their own section walls (no overlap possible across
    sections), and within-section grid layouts rarely require
    overlap-avoidance for n_per_section <= 4 (the synthetic shop's
    design). If the optimum places items on top of each other within a
    section, that's a real result the GA must also avoid -- and the GA's
    fitness function does explicitly penalize within-section overlap.
    """
    ordered_names = [it.name for it in shop.items]
    bounds = _bounds_for(shop, ordered_names)
    bounds_arr = np.asarray(bounds, dtype=np.float64)

    def neg_R(vec: np.ndarray) -> float:
        layout = _vec_to_layout(vec, ordered_names)
        return -analytical_revenue(shop, layout)

    best_vec = None
    best_val = -np.inf
    rng = np.random.default_rng(shop.rng_seed)
    starts: List[np.ndarray] = []
    starts.append(_layout_to_vec(grid_layout_within_sections(shop), ordered_names))
    for _ in range(n_restarts - 1):
        rand_vec = bounds_arr[:, 0] + rng.random(len(bounds)) * (
            bounds_arr[:, 1] - bounds_arr[:, 0]
        )
        starts.append(rand_vec)

    total_iter = 0
    converged_any = False
    for x0 in starts:
        try:
            res = minimize(neg_R, x0, method=method, bounds=bounds)
        except Exception:
            continue
        total_iter += int(getattr(res, "nit", 0))
        if not np.isfinite(res.fun):
            continue
        if -res.fun > best_val:
            best_val = float(-res.fun)
            best_vec = res.x
            converged_any = converged_any or bool(res.success)

    if best_vec is None:
        # Pathological: fall back to grid layout. Should not happen on
        # well-formed synthetic shops.
        layout = grid_layout_within_sections(shop)
        return OracleResult(
            layout=layout,
            true_revenue=analytical_revenue(shop, layout),
            converged=False, n_iter=0, method=method,
            note="all restarts failed; falling back to grid layout",
        )

    return OracleResult(
        layout=_vec_to_layout(best_vec, ordered_names),
        true_revenue=best_val,
        converged=converged_any,
        n_iter=total_iter,
        method=method,
    )


def test_oracle_anti_circularity() -> bool:
    """Sanity check: this module must not transitively import the
    simulator or GA. Raises AssertionError on violation."""
    import sys
    banned = {"viz_ga", "viz_ga_run", "viz_optimize", "viz_optimize_helpers",
              "viz_optimize_results", "simulation", "customer",
              "customer_pathfinding"}
    leaked = banned & set(sys.modules)
    assert not leaked, f"oracle.py transitively imports simulator: {leaked}"
    return True


if __name__ == "__main__":
    from synthetic_shops import generate_synthetic_shop
    for label, n_items, w, h in [("toy6", 6, 8.0, 6.0),
                                  ("std10", 10, 12.0, 10.0),
                                  ("std12", 12, 14.0, 11.0)]:
        shop = generate_synthetic_shop(label, seed=42, n_items=n_items,
                                       width=w, height=h)
        impulse_names = [i.name for i in shop.items if i.is_impulse]
        print(f"\n{'='*60}")
        print(f"Shop: {label}  ({len(shop.items)} items, "
              f"{len(shop.sections)} sections, "
              f"{len(impulse_names)} impulse: {impulse_names})")
        naive = grid_layout_within_sections(shop)
        naive_R = analytical_revenue(shop, naive)
        result = solve_oracle(shop, n_restarts=8)
        lift = (result.true_revenue - naive_R) / max(naive_R, 1e-6) * 100
        print(f"  Grid-layout R = {naive_R:>12.2f}")
        print(f"  Oracle R      = {result.true_revenue:>12.2f}  "
              f"(converged={result.converged}, n_iter={result.n_iter})")
        print(f"  Oracle lift over grid: {lift:+.2f}%")
        # Sanity: impulse items should end up closer to checkout under
        # the oracle than under the grid layout.
        if impulse_names:
            chk = shop.checkout
            grid_dist = sum(math.hypot(naive[n][0] - chk[0],
                                       naive[n][1] - chk[1])
                            for n in impulse_names) / len(impulse_names)
            orac_dist = sum(math.hypot(result.layout[n][0] - chk[0],
                                       result.layout[n][1] - chk[1])
                            for n in impulse_names) / len(impulse_names)
            print(f"  Impulse mean dist to checkout: "
                  f"grid={grid_dist:.2f}m -> oracle={orac_dist:.2f}m "
                  f"({(orac_dist - grid_dist) / max(grid_dist, 1e-6) * 100:+.1f}%)")
    test_oracle_anti_circularity()
    print("\nAnti-circularity check passed (oracle does not import simulator).")
