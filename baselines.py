"""Layout baselines for the GA-vs-baselines paper figure.

Each baseline takes a ``SyntheticShop`` and returns a dict
``{item_name: (x, y)}`` -- the same shape ``analytical_revenue`` and the
simulator-backed fitness function both consume. All baselines respect
the same section-bound + non-overlap constraints the GA enforces via
``viz_ga_run._ga_repair``; this is critical to avoid the strawman-random
trap where 'random' looks absurd because items can overlap freely.

Four baselines, ordered roughly by sophistication:

  1. ``random_valid``   - uniform random within section bounds,
                          then collision-resolved by greedy nudging.
  2. ``perimeter_only`` - Larson 2005 rule: place every item as close
                          to the section's perimeter (and thus the
                          shop's perimeter) as possible.
  3. ``popularity_rank``- items ranked by base_revenue placed in the
                          'best' position within their section
                          (closest to entrance + section centre).
                          Mirrors what dataset_layout.py does for
                          real-data shops.
  4. ``greedy_swap``    - start from popularity_rank, then for each
                          pair (i, j) try swapping positions; keep
                          the swap if analytical_revenue increases.
                          Single pass, O(N^2) evaluations.

The GA's job is to beat all four under paired-MC comparison.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from synthetic_shops import (
    SyntheticShop,
    SyntheticItem,
    analytical_revenue,
    grid_layout_within_sections,
)


# ─── Constraint helpers ───────────────────────────────────────────────────

def _section_inner_bounds(shop: SyntheticShop,
                          item: SyntheticItem,
                          pad: float = 0.05
                          ) -> Tuple[float, float, float, float]:
    """(lo_x, lo_y, hi_x, hi_y) for the item's top-left position so the
    full item rectangle fits inside its section with ``pad`` margin."""
    sec = shop.sections.get(item.category)
    iw, ih = item.size
    if sec is None:
        return (pad, pad,
                max(pad, shop.width - iw - pad),
                max(pad, shop.height - ih - pad))
    sx, sy, sw, sh = sec
    return (sx + pad, sy + pad,
            max(sx + pad, sx + sw - iw - pad),
            max(sy + pad, sy + sh - ih - pad))


def _items_overlap(a: Tuple[float, float], a_sz: Tuple[float, float],
                   b: Tuple[float, float], b_sz: Tuple[float, float]) -> bool:
    ax, ay = a; aw, ah = a_sz
    bx, by = b; bw, bh = b_sz
    return not (ax + aw <= bx or ax >= bx + bw or
                ay + ah <= by or ay >= by + bh)


def _repair_within_section(shop: SyntheticShop,
                            layout: Dict[str, Tuple[float, float]],
                            max_passes: int = 10) -> Dict[str, Tuple[float, float]]:
    """Resolve within-section overlaps by spreading items deterministically
    along the section's longer axis. Items are sorted by current position
    on the spread axis, then re-placed at evenly-spaced slots that
    guarantee non-overlap (slot step = max_item_dim + gap).

    Section-bound clipping is preserved. If the section is too small to
    fit its items even at the tight spacing, residual overlaps remain
    (caller's responsibility — generate_synthetic_shop sizes sections to
    avoid this)."""
    by_name = {it.name: it for it in shop.items}
    by_cat: Dict[str, List[str]] = {}
    for name in layout:
        by_cat.setdefault(by_name[name].category, []).append(name)

    for cat, names in by_cat.items():
        if len(names) < 2:
            continue
        sec = shop.sections.get(cat)
        if sec is None:
            continue
        sx, sy, sw, sh = sec
        pad = 0.05
        gap = 0.20

        # Detect any pair-wise overlap; if none, skip section.
        any_overlap = False
        for i in range(len(names)):
            ni = names[i]; pi = layout[ni]; si = by_name[ni].size
            for j in range(i + 1, len(names)):
                nj = names[j]; pj = layout[nj]; sj = by_name[nj].size
                if _items_overlap(pi, si, pj, sj):
                    any_overlap = True
                    break
            if any_overlap:
                break
        if not any_overlap:
            continue

        # Spread along the longer section axis. For each item, slot
        # step = max dim of any item in this section + gap.
        max_w = max(by_name[n].size[0] for n in names)
        max_h = max(by_name[n].size[1] for n in names)
        if sw >= sh:
            # Spread horizontally; preserve vertical order.
            step = max_w + gap
            usable = sw - 2 * pad - max_w
            n = len(names)
            if n > 1:
                actual_step = max(step, usable / (n - 1))
            else:
                actual_step = step
            # Sort by current x so adjacency is preserved as best we can.
            ordered = sorted(names, key=lambda nm: layout[nm][0])
            for k, nm in enumerate(ordered):
                iw, ih = by_name[nm].size
                x = sx + pad + k * actual_step
                # Clip to section bounds even if shop didn't allocate
                # enough width (residual overlap acceptable).
                x = max(sx + pad, min(x, sx + sw - iw - pad))
                # Keep y, but clip to section bounds.
                y = layout[nm][1]
                y = max(sy + pad, min(y, sy + sh - ih - pad))
                layout[nm] = (float(x), float(y))
        else:
            step = max_h + gap
            usable = sh - 2 * pad - max_h
            n = len(names)
            actual_step = max(step, usable / max(n - 1, 1))
            ordered = sorted(names, key=lambda nm: layout[nm][1])
            for k, nm in enumerate(ordered):
                iw, ih = by_name[nm].size
                y = sy + pad + k * actual_step
                y = max(sy + pad, min(y, sy + sh - ih - pad))
                x = layout[nm][0]
                x = max(sx + pad, min(x, sx + sw - iw - pad))
                layout[nm] = (float(x), float(y))
    return layout


# ─── Baselines ────────────────────────────────────────────────────────────

def random_valid(shop: SyntheticShop, seed: int = 0
                 ) -> Dict[str, Tuple[float, float]]:
    """Uniform random per-item position within section bounds, then
    overlap-repaired. NOT a strawman: it respects the same section
    constraint the GA does."""
    rng = np.random.default_rng(seed)
    layout: Dict[str, Tuple[float, float]] = {}
    for it in shop.items:
        lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(shop, it)
        layout[it.name] = (
            float(rng.uniform(lo_x, hi_x)),
            float(rng.uniform(lo_y, hi_y)),
        )
    np.random.seed(seed)   # _repair_within_section uses np.random
    return _repair_within_section(shop, layout)


def perimeter_only(shop: SyntheticShop, seed: int = 0
                   ) -> Dict[str, Tuple[float, float]]:
    """Place every item as close to its section's perimeter as possible.
    Larson 2005 racetrack hypothesis: perimeter gets more traffic.

    Implementation: for each item, pick the section corner closest to
    the section's nearest outer wall edge. Multiple items per section
    are spread along the perimeter."""
    by_cat: Dict[str, List[SyntheticItem]] = {}
    for it in shop.items:
        by_cat.setdefault(it.category, []).append(it)
    layout: Dict[str, Tuple[float, float]] = {}
    for cat, items in by_cat.items():
        sec = shop.sections.get(cat)
        if sec is None:
            continue
        sx, sy, sw, sh = sec
        # Distance from each section edge to nearest shop wall.
        edge_dists = {
            'left':   sx,
            'right':  shop.width - (sx + sw),
            'bottom': sy,
            'top':    shop.height - (sy + sh),
        }
        # Sort edges by closeness to a shop wall (ascending).
        ranked_edges = sorted(edge_dists.items(), key=lambda kv: kv[1])
        for k, it in enumerate(items):
            iw, ih = it.size
            edge_name, _ = ranked_edges[k % len(ranked_edges)]
            # Spread along the chosen edge
            n_on_edge = max(1, sum(1 for j, _ in enumerate(items)
                                    if ranked_edges[j % len(ranked_edges)][0] == edge_name))
            slot = k // len(ranked_edges)
            if edge_name in ('bottom', 'top'):
                spacing = max((sw - iw - 0.2) / max(n_on_edge, 1), iw + 0.1)
                x = sx + 0.1 + slot * spacing
                y = sy + 0.05 if edge_name == 'bottom' else sy + sh - ih - 0.05
            else:
                spacing = max((sh - ih - 0.2) / max(n_on_edge, 1), ih + 0.1)
                x = sx + 0.05 if edge_name == 'left' else sx + sw - iw - 0.05
                y = sy + 0.1 + slot * spacing
            lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(shop, it)
            x = float(np.clip(x, lo_x, hi_x))
            y = float(np.clip(y, lo_y, hi_y))
            layout[it.name] = (x, y)
    np.random.seed(seed)
    return _repair_within_section(shop, layout)


def popularity_rank(shop: SyntheticShop, seed: int = 0
                    ) -> Dict[str, Tuple[float, float]]:
    """Items ranked by base_revenue placed at the 'best' position within
    their section, where 'best' = closest to (entrance, section centre).

    This is the closest thing to what ``dataset_layout`` does for
    real-data shops -- the popular-items-front-and-centre heuristic
    practitioners reach for first. A strong baseline.

    Slot spacing is computed from the items' actual sizes so adjacent
    slots cannot overlap by construction."""
    by_cat: Dict[str, List[SyntheticItem]] = {}
    for it in shop.items:
        by_cat.setdefault(it.category, []).append(it)
    layout: Dict[str, Tuple[float, float]] = {}
    for cat, items in by_cat.items():
        items_sorted = sorted(items, key=lambda it: it.base_revenue, reverse=True)
        sec = shop.sections.get(cat)
        if sec is None:
            continue
        sx, sy, sw, sh = sec
        n = len(items_sorted)
        cols = max(1, min(2, n))
        rows = int(math.ceil(n / cols))
        # Slot size = max item dim + gap; this guarantees no overlap.
        max_iw = max(it.size[0] for it in items_sorted)
        max_ih = max(it.size[1] for it in items_sorted)
        pad = 0.20
        gap = 0.25
        # Available width/height for the grid
        avail_w = sw - 2 * pad
        avail_h = sh - 2 * pad
        # Step must be >= max_iw + gap (or fit exactly if cols=1)
        step_x = max(max_iw + gap, avail_w / max(cols, 1))
        step_y = max(max_ih + gap, avail_h / max(rows, 1))

        cx_sec = sx + sw / 2
        slot_scores: List[Tuple[float, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                slot_x = sx + pad + c * step_x
                slot_y = sy + pad + r * step_y
                score = abs(slot_x + max_iw / 2 - cx_sec) + 2.0 * (slot_y - sy)
                slot_scores.append((score, r, c))
        slot_scores.sort()
        for idx, it in enumerate(items_sorted):
            if idx >= len(slot_scores):
                break
            _, r, c = slot_scores[idx]
            x = sx + pad + c * step_x
            y = sy + pad + r * step_y
            lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(shop, it)
            layout[it.name] = (float(np.clip(x, lo_x, hi_x)),
                               float(np.clip(y, lo_y, hi_y)))
    np.random.seed(seed)
    return _repair_within_section(shop, layout)


def greedy_swap(shop: SyntheticShop, seed: int = 0,
                max_swaps: int = None) -> Dict[str, Tuple[float, float]]:
    """Start from popularity_rank; for each pair of items in the same
    section, try swapping their positions; keep the swap iff
    analytical_revenue strictly increases. Single pass over all pairs.

    Crucially: swaps are constrained to within-section pairs (cross-
    section swaps would violate the section-bound constraint). This is
    a cheap local-search baseline; the GA's job is to beat it because
    it can move items *continuously* within sections rather than only
    swap among grid slots."""
    layout = popularity_rank(shop, seed=seed)
    by_cat: Dict[str, List[str]] = {}
    by_name = {it.name: it for it in shop.items}
    for name in layout:
        by_cat.setdefault(by_name[name].category, []).append(name)
    swaps_done = 0
    cap = max_swaps if max_swaps is not None else 10 * len(shop.items)
    cur_R = analytical_revenue(shop, layout)
    for cat, names in by_cat.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if swaps_done >= cap:
                    break
                ni, nj = names[i], names[j]
                pi, pj = layout[ni], layout[nj]
                # Need to clamp to each item's own section bounds (same
                # category so same section; only the size differs).
                lo_x_i, lo_y_i, hi_x_i, hi_y_i = _section_inner_bounds(shop, by_name[ni])
                lo_x_j, lo_y_j, hi_x_j, hi_y_j = _section_inner_bounds(shop, by_name[nj])
                new_pi = (float(np.clip(pj[0], lo_x_i, hi_x_i)),
                          float(np.clip(pj[1], lo_y_i, hi_y_i)))
                new_pj = (float(np.clip(pi[0], lo_x_j, hi_x_j)),
                          float(np.clip(pi[1], lo_y_j, hi_y_j)))
                trial = dict(layout)
                trial[ni] = new_pi
                trial[nj] = new_pj
                trial_R = analytical_revenue(shop, trial)
                if trial_R > cur_R:
                    layout = trial
                    cur_R = trial_R
                    swaps_done += 1
    return layout


# ─── Validation helpers (callable from experiments) ───────────────────────

def assert_layout_valid(shop: SyntheticShop,
                        layout: Dict[str, Tuple[float, float]],
                        tol: float = 0.06) -> None:
    """Raises AssertionError if the layout violates section bounds or
    has within-section overlaps. ``tol`` allows for the 0.05m padding
    plus a small rounding fudge."""
    by_name = {it.name: it for it in shop.items}
    # Section-bound check
    for name, (x, y) in layout.items():
        it = by_name[name]
        sec = shop.sections.get(it.category)
        if sec is None:
            continue
        sx, sy, sw, sh = sec
        iw, ih = it.size
        if not (sx - tol <= x <= sx + sw - iw + tol and
                sy - tol <= y <= sy + sh - ih + tol):
            raise AssertionError(
                f"{name} ({it.category}) at ({x:.2f},{y:.2f}) outside "
                f"section bounds ({sx},{sy})-{sw}x{sh}")
    # Within-section overlap check
    by_cat: Dict[str, List[str]] = {}
    for name in layout:
        by_cat.setdefault(by_name[name].category, []).append(name)
    for cat, names in by_cat.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ni, nj = names[i], names[j]
                pi, si = layout[ni], by_name[ni].size
                pj, sj = layout[nj], by_name[nj].size
                if _items_overlap(pi, si, pj, sj):
                    # Allow tiny overlaps (numerical residue from repair)
                    overlap_x = min(pi[0] + si[0], pj[0] + sj[0]) - max(pi[0], pj[0])
                    overlap_y = min(pi[1] + si[1], pj[1] + sj[1]) - max(pi[1], pj[1])
                    if overlap_x > tol and overlap_y > tol:
                        raise AssertionError(
                            f"{ni} and {nj} overlap in section {cat}: "
                            f"({overlap_x:.3f}, {overlap_y:.3f})")


if __name__ == "__main__":
    # Smoke test: every baseline produces a valid layout on a std10 shop
    # and analytical_revenue ranks them in a plausible order.
    from synthetic_shops import generate_synthetic_shop
    shop = generate_synthetic_shop("std10", seed=42, n_items=10,
                                   width=12.0, height=10.0)
    grid_R = analytical_revenue(shop, grid_layout_within_sections(shop))

    methods = [
        ("random_valid",    random_valid),
        ("perimeter_only",  perimeter_only),
        ("popularity_rank", popularity_rank),
        ("greedy_swap",     greedy_swap),
    ]
    print(f"Shop std10: {len(shop.items)} items; grid R = {grid_R:.2f}")
    print()
    for name, fn in methods:
        layout = fn(shop, seed=0)
        try:
            assert_layout_valid(shop, layout)
            status = "valid"
        except AssertionError as e:
            status = f"INVALID: {e}"
        R = analytical_revenue(shop, layout)
        lift = (R - grid_R) / grid_R * 100
        print(f"  {name:18s}  R = {R:>10.2f}  lift_vs_grid={lift:+6.2f}%  [{status}]")
