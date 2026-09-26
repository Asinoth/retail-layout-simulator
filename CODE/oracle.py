"""Analytical reference solution for a SyntheticShop.

Works strictly off the closed-form revenue model in
synthetic_shops.analytical_revenue. The GA's fitness is a different
function of the layout: ``viz_ga._ga_compute_layout_score`` scores the
layout against a painted cold-start heat map (the synthetic prior of
``experiments._common.paint_synthetic_heatmap``), co-purchase distances,
an entrance-to-checkout waypoint path and the checkout's distance, and the
Monte Carlo engine turns that score into revenue. No agent walks and no
live queue forms in either; the two share the literature directions and the
physical validity rule (no overlap), not a functional form. Any ranking
agreement between them is therefore informative rather than circular.

The anti-circularity guarantee is the whole point: this module imports
neither viz_ga nor viz_optimize nor simulation; its only project
dependency is synthetic_shops. ``tests/test_oracle.py`` checks it in a
fresh interpreter, where no other test can have imported them first.

The code and the shipped CSVs keep the historical name ``oracle``; the
paper calls the result the analytical reference solution. It is the best
of a multi-start local search, not a certified global optimum:
``OracleResult.best_curve`` records how the best value grew with the
number of restarts, so a reader can see whether it had stopped growing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    """``layout`` and its ``true_revenue`` (analytical, after contact
    separation); ``converged`` is True when some restart reported success
    and the layout has no residual overlap. ``n_restarts`` starts were
    tried, ``n_failed`` of them failed (``failures``: ``(restart, reason)``);
    ``restart_values`` holds each restart's solver value (None for a
    failed one) and ``best_curve[k]`` the best of the first k + 1."""
    layout: Dict[str, Tuple[float, float]]
    true_revenue: float
    converged: bool
    n_iter: int
    method: str
    note: str = ""
    n_restarts: int = 0
    n_failed: int = 0
    failures: List[Tuple[int, str]] = field(default_factory=list)
    restart_values: List[Optional[float]] = field(default_factory=list)
    best_curve: List[Optional[float]] = field(default_factory=list)


class OracleFailure(RuntimeError):
    """Every restart of the analytical reference failed."""


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


# Gap left between two items separated after the solve. Large against
# float rounding at shop scale (~1e-15 m), negligible for revenue.
_SEPARATION_CLEARANCE = 1e-6


def _separate_contacts(shop: SyntheticShop,
                       ordered_names: List[str],
                       vec: np.ndarray,
                       bounds_arr: np.ndarray,
                       max_passes: int = 20) -> np.ndarray:
    """Remove residual same-section overlaps from a solver vector.

    L-BFGS-B pushes neighbouring items flush and can stop with an overlap
    of ~1e-8 m. ``analytical_revenue`` penalizes overlap by area, so that
    residue costs nothing here, but the simulator fitness counts ANY
    positive overlap as a full penalty. For each pair with ox > 0 and
    oy > 0, the later item moves along the axis of smaller overlap until
    it clears the earlier one by ``_SEPARATION_CLEARANCE``; if its bound
    stops it, the earlier item moves the opposite way by the shortfall.
    The other axis is tried only when neither item can make room on the
    first. Repeats for at most ``max_passes`` passes, since a move can
    create a new contact within a section."""
    vec = np.array(vec, dtype=np.float64, copy=True)
    by_name = {it.name: it for it in shop.items}
    sizes = [by_name[n].size for n in ordered_names]
    idx_by_cat: Dict[str, List[int]] = {}
    for i, name in enumerate(ordered_names):
        idx_by_cat.setdefault(by_name[name].category, []).append(i)

    def _push_apart(i: int, j: int, ax: int) -> bool:
        di, dj = 2 * i + ax, 2 * j + ax
        pi, pj = vec[di], vec[dj]
        si, sj = sizes[i][ax], sizes[j][ax]
        lo_i, hi_i = bounds_arr[di]
        lo_j, hi_j = bounds_arr[dj]
        new_pi = pi
        if pj + sj / 2.0 >= pi + si / 2.0:
            # j sits on the high side of i.
            target_j = pi + si + _SEPARATION_CLEARANCE
            new_pj = min(target_j, hi_j)
            if new_pj < target_j:
                new_pi = max(new_pj - si - _SEPARATION_CLEARANCE, lo_i)
            separated = new_pi + si <= new_pj
        else:
            target_j = pi - sj - _SEPARATION_CLEARANCE
            new_pj = max(target_j, lo_j)
            if new_pj > target_j:
                new_pi = min(new_pj + sj + _SEPARATION_CLEARANCE, hi_i)
            separated = new_pj + sj <= new_pi
        if separated:
            vec[di], vec[dj] = new_pi, new_pj
        return separated

    for _ in range(max_passes):
        moved = False
        for idx in idx_by_cat.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    xi, yi = vec[2 * i], vec[2 * i + 1]
                    xj, yj = vec[2 * j], vec[2 * j + 1]
                    (wi, hi), (wj, hj) = sizes[i], sizes[j]
                    ox = min(xi + wi, xj + wj) - max(xi, xj)
                    oy = min(yi + hi, yj + hj) - max(yi, yj)
                    if not (ox > 0 and oy > 0):
                        continue
                    axes = (0, 1) if ox <= oy else (1, 0)
                    if any(_push_apart(i, j, ax) for ax in axes):
                        moved = True
        if not moved:
            break
    return vec


def _strict_overlap_pairs(shop: SyntheticShop,
                          ordered_names: List[str],
                          vec: np.ndarray) -> List[Tuple[str, str]]:
    """Same-section pairs that still overlap with positive area.

    ``_separate_contacts`` gives up after its passes, or as soon as no
    single push succeeds, so it can return a vector that still overlaps;
    the simulator fitness charges a full penalty for any positive overlap,
    and the repair the experiment harness applies before scoring would
    silently replace such a layout with a grid pack. Checking here lets the
    caller report that the reference solution is not the one it proposed."""
    by_name = {it.name: it for it in shop.items}
    idx_by_cat: Dict[str, List[int]] = {}
    for i, name in enumerate(ordered_names):
        idx_by_cat.setdefault(by_name[name].category, []).append(i)
    out: List[Tuple[str, str]] = []
    for idx in idx_by_cat.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                xi, yi = vec[2 * i], vec[2 * i + 1]
                xj, yj = vec[2 * j], vec[2 * j + 1]
                (wi, hi) = by_name[ordered_names[i]].size
                (wj, hj) = by_name[ordered_names[j]].size
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if ox > 0 and oy > 0:
                    out.append((ordered_names[i], ordered_names[j]))
    return out


def solve_oracle(shop: SyntheticShop,
                 n_restarts: int = 6,
                 method: str = "L-BFGS-B") -> OracleResult:
    """Solve max R(layout) over per-item-section bounds.

    Uses multi-start L-BFGS-B (gradient by finite differences). Initial
    points: the grid layout + (n_restarts - 1) uniform draws within the
    bounds, from a generator seeded by the shop's own seed.

    The objective is piecewise smooth, not smooth: ``analytical_revenue``
    has kinks where an item's nearest wall changes (the min in d_wall),
    where the perimeter term is floored at zero, where an item's revenue is
    floored at zero, at the entrance, checkout and partner-item centres
    (the Euclidean distances), and where two items start to overlap (the
    area penalty). L-BFGS-B converges to a local optimum of each smooth
    piece it lands in, which is why the search restarts from many points;
    ``best_curve`` shows how the best value grew with the restarts.

    Non-overlap is NOT a constraint in the optimization: items are
    confined to their own section walls (no overlap possible across
    sections), and within-section overlap is priced by the area penalty
    of ``analytical_revenue``. If the optimum places items on top of each
    other within a section, that's a real result the GA must also avoid --
    and the GA's fitness function does explicitly penalize within-section
    overlap.

    A restart whose solver raises, or returns a non-finite value, is
    recorded in ``failures`` with its reason rather than skipped silently;
    ``OracleFailure`` is raised when every restart fails, instead of
    handing back a fallback layout as if it were a solution.

    The best vector is then cleared of contact residue by
    ``_separate_contacts`` (the solver leaves flush neighbours overlapping
    by ~1e-8 m, which the simulator fitness treats as a real overlap),
    and ``true_revenue`` is recomputed on that separated layout. Separation
    can fail in a crowded section, so the result is checked afterwards and
    reported as not converged, with the offending pairs in ``note``, rather
    than handed on as a clean reference solution.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
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
    failures: List[Tuple[int, str]] = []
    values: List[Optional[float]] = []
    curve: List[Optional[float]] = []
    for k, x0 in enumerate(starts):
        try:
            res = minimize(neg_R, x0, method=method, bounds=bounds)
        except Exception as exc:          # recorded, never swallowed
            failures.append((k, f"{type(exc).__name__}: {exc}"))
            values.append(None)
            curve.append(None if best_vec is None else best_val)
            continue
        total_iter += int(getattr(res, "nit", 0))
        if not np.isfinite(res.fun):
            failures.append((k, f"non-finite objective {res.fun!r}"))
            values.append(None)
            curve.append(None if best_vec is None else best_val)
            continue
        values.append(float(-res.fun))
        if -res.fun > best_val:
            best_val = float(-res.fun)
            best_vec = res.x
            converged_any = converged_any or bool(res.success)
        curve.append(best_val)

    if best_vec is None:
        raise OracleFailure(
            f"all {len(starts)} restarts of the analytical reference failed "
            f"on {shop.name}: " + '; '.join(f"#{k}: {r}"
                                           for k, r in failures[:5]))

    sep_vec = _separate_contacts(shop, ordered_names, best_vec, bounds_arr)
    layout = _vec_to_layout(sep_vec, ordered_names)
    residual = _strict_overlap_pairs(shop, ordered_names, sep_vec)
    note = ""
    if residual:
        note = ("residual overlap after separation: "
                + ", ".join(f"{a}/{b}" for a, b in residual[:5])
                + ("" if len(residual) <= 5
                   else f" (+{len(residual) - 5} more)"))
    return OracleResult(
        layout=layout,
        true_revenue=analytical_revenue(shop, layout),
        converged=converged_any and not residual,
        n_iter=total_iter,
        method=method,
        note=note,
        n_restarts=len(starts),
        n_failed=len(failures),
        failures=failures,
        restart_values=values,
        best_curve=curve,
    )


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
        if result.note:
            print(f"  note: {result.note}")
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
    print("\nThe anti-circularity check runs in tests/test_oracle.py.")
