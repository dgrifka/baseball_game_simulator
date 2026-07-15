"""
Vectorized game-simulation engine.

Simulates all N games at once as NumPy arrays, iterating over event POSITIONS
(~80 per team) instead of over simulations (10,000). The scalar functions in
game_simulator.py (simulate_game / advance_runner / attempt_steal /
attempt_pickoff) remain the reference implementation: every transition table
here is built at import time BY CALLING them, with the single stochastic
branch draw forced each way via a temporarily patched random.random. There is
deliberately no re-implementation of baserunning logic in this file — if the
scalar rules change, the tables follow automatically.

State encoding: bases as int 0-7 (bit 0 = 1st, bit 1 = 2nd, bit 2 = 3rd).
Every event consumes at most ONE branch draw (the three probabilistic
advancement rules in advance_runner are mutually exclusive per state and hit
type), so each event resolves to: classify (for batted balls), then one table
lookup on (state, class, branch-bit).

Design/validation: discussed in the private orchestration repo's documentation — exhaustive
table-equivalence tests plus a null-calibrated 200-game replay gate.
"""

import numpy as np

# Event codes shared by the translators and the sim loops.
EV_STRIKEOUT = 0
EV_WALK = 1
EV_STEAL = 2
EV_PICKOFF = 3
EV_BBE = 4

_N_STATES = 8  # bases bitmask 0-7


def _bases_from_state(s):
    return [bool(s & 1), bool(s & 2), bool(s & 4)]


def _state_from_bases(bases):
    return int(bases[0]) | (int(bases[1]) << 1) | (int(bases[2]) << 2)


def build_transition_tables(advance_runner, attempt_steal, attempt_pickoff,
                            branch_probs):
    """Build all lookup tables from the scalar reference functions.

    Args:
        advance_runner, attempt_steal, attempt_pickoff: the scalar functions
            from game_simulator (passed in to avoid a circular import).
        branch_probs (dict): {(state, count): p} for the transitions whose
            branch draw matters — derived from the SINGLE_ADV_*/DOUBLE_ADV_*
            constants; consulted only to fill BRANCH_P below.

    Returns dict of np arrays:
        WALK_NEXT[8], WALK_RUNS[8]
        STEAL_NEXT[8], STEAL_RUNS[8]          (state 0 = identity/no-op)
        PICK_NEXT[8], PICK_OUTS[8]            (state 0 = no-op, no out)
        HIT_NEXT[2, 8, 5], HIT_RUNS[2, 8, 5]  (branch, state, count 1-4;
                                               branch 1 = draw < p taken;
                                               deterministic entries have
                                               branch 0 == branch 1)
        BRANCH_P[8, 5]                        (0.0 where deterministic)
    """
    import random as _random

    walk_next = np.zeros(_N_STATES, dtype=np.int8)
    walk_runs = np.zeros(_N_STATES, dtype=np.int16)
    steal_next = np.zeros(_N_STATES, dtype=np.int8)
    steal_runs = np.zeros(_N_STATES, dtype=np.int16)
    pick_next = np.zeros(_N_STATES, dtype=np.int8)
    pick_outs = np.zeros(_N_STATES, dtype=np.int8)
    hit_next = np.zeros((2, _N_STATES, 5), dtype=np.int8)
    hit_runs = np.zeros((2, _N_STATES, 5), dtype=np.int16)
    branch_p = np.zeros((_N_STATES, 5), dtype=np.float64)

    def _forced(fn, forced_value):
        """Run fn with random.random pinned to forced_value."""
        orig = _random.random
        _random.random = lambda: forced_value
        try:
            return fn()
        finally:
            _random.random = orig

    for s in range(_N_STATES):
        # Walk (no randomness)
        b = _bases_from_state(s)
        walk_runs[s] = advance_runner(b, is_walk=True)
        walk_next[s] = _state_from_bases(b)

        # Steal / pickoff: the sim loop only applies them when any(bases);
        # state 0 rows are identity no-ops and never selected.
        if s:
            b = _bases_from_state(s)
            steal_runs[s] = attempt_steal(b)
            steal_next[s] = _state_from_bases(b)
            b = _bases_from_state(s)
            pick_outs[s] = attempt_pickoff(b)
            pick_next[s] = _state_from_bases(b)

        # Hits: force the single branch draw both ways.
        for count in range(1, 5):
            for branch, forced in ((0, 0.999999999), (1, 0.0)):
                b = _bases_from_state(s)
                runs = _forced(lambda: advance_runner(b, count=count), forced)
                hit_runs[branch, s, count] = runs
                hit_next[branch, s, count] = _state_from_bases(b)
            if (hit_next[0, s, count] != hit_next[1, s, count]
                    or hit_runs[0, s, count] != hit_runs[1, s, count]):
                branch_p[s, count] = branch_probs[(s, count)]

    # Unified deterministic tables indexed [event_code, state] so the sim loop
    # resolves every non-BBE event in ONE gather. The EV_BBE row is identity /
    # zero — the BBE block overrides it.
    det_next = np.zeros((5, _N_STATES), dtype=np.int8)
    det_runs = np.zeros((5, _N_STATES), dtype=np.int16)
    det_outs = np.zeros((5, _N_STATES), dtype=np.int8)
    ident = np.arange(_N_STATES, dtype=np.int8)
    det_next[EV_STRIKEOUT] = ident
    det_outs[EV_STRIKEOUT] = 1
    det_next[EV_WALK] = walk_next
    det_runs[EV_WALK] = walk_runs
    det_next[EV_STEAL] = steal_next
    det_runs[EV_STEAL] = steal_runs
    det_next[EV_PICKOFF] = pick_next
    det_outs[EV_PICKOFF] = pick_outs
    det_next[EV_BBE] = ident

    return {
        "WALK_NEXT": walk_next, "WALK_RUNS": walk_runs,
        "STEAL_NEXT": steal_next, "STEAL_RUNS": steal_runs,
        "PICK_NEXT": pick_next, "PICK_OUTS": pick_outs,
        "HIT_NEXT": hit_next, "HIT_RUNS": hit_runs,
        "BRANCH_P": branch_p,
        "DET_NEXT": det_next, "DET_RUNS": det_runs, "DET_OUTS": det_outs,
    }


def translate_outcomes(outcomes_clean, prob_cache, cache_key_fn, innings=None):
    """Turn a cleaned outcome list into (event_codes, cdf[, innings]) arrays.

    Batted balls whose key misses prob_cache are DROPPED here — the scalar
    loop `continue`s them mid-game, which is distributionally identical
    (the relative order of the remaining events stays uniform under the
    per-sim permutation). Unrecognized outcomes are skipped, as in the
    scalar loops.

    Args:
        innings: optional sequence aligned with outcomes_clean (1-based
            inning per event). When given, entries are filtered in lockstep
            and returned as a third int64 array.

    Returns:
        ev_code: int8 (n_events,)
        cdf:     float64 (n_events, 4) cumulative P(out), P(≤1B), P(≤2B),
                 P(≤3B); rows for non-BBE events are zero and unused.
        innings_out (only when innings is not None): int64 (n_events,)
    """
    codes, cdfs, inns = [], [], []
    zero = (0.0, 0.0, 0.0, 0.0)
    for i, outcome in enumerate(outcomes_clean):
        if outcome == "strikeout":
            codes.append(EV_STRIKEOUT); cdfs.append(zero)
        elif outcome == "walk":
            codes.append(EV_WALK); cdfs.append(zero)
        elif outcome == "stolen_base":
            codes.append(EV_STEAL); cdfs.append(zero)
        elif outcome == "pickoff":
            codes.append(EV_PICKOFF); cdfs.append(zero)
        elif isinstance(outcome, (dict, tuple)):
            key = cache_key_fn(outcome) if isinstance(outcome, dict) else outcome
            p = prob_cache.get(key)
            if p is None:
                continue
            c0 = float(p[0]); c1 = c0 + float(p[1])
            c2 = c1 + float(p[2]); c3 = c2 + float(p[3])
            codes.append(EV_BBE); cdfs.append((c0, c1, c2, c3))
        else:
            continue
        if innings is not None:
            inns.append(innings[i])
    out = (np.asarray(codes, dtype=np.int8),
           np.asarray(cdfs, dtype=np.float64).reshape(len(codes), 4))
    if innings is not None:
        return out + (np.asarray(inns, dtype=np.int64),)
    return out


def simulate_games_vectorized(ev_code, cdf, tables, num_simulations, rng):
    """Vectorized equivalent of num_simulations calls to simulate_game().

    Args:
        ev_code, cdf: from translate_outcomes().
        tables: from build_transition_tables().
        num_simulations (int), rng (np.random.Generator).

    Returns:
        np.ndarray (num_simulations,) int32 — runs per simulated game.
    """
    n_ev = len(ev_code)
    runs = np.zeros(num_simulations, dtype=np.int32)
    if n_ev == 0:
        return runs

    # Per-sim uniform random permutation of the event order (Fisher-Yates per
    # column via rng.permuted — ~2x cheaper than argsort-of-uniforms).
    # Transposed layout — (n_events, n_sims) — so each loop step reads a
    # contiguous row.
    order = rng.permuted(
        np.tile(np.arange(n_ev, dtype=np.int16)[:, None], (1, num_simulations)),
        axis=0)
    ev_seq = np.ascontiguousarray(ev_code[order])

    bases = np.zeros(num_simulations, dtype=np.int8)
    outs = np.zeros(num_simulations, dtype=np.int8)

    det_next, det_runs, det_outs = (tables["DET_NEXT"], tables["DET_RUNS"],
                                    tables["DET_OUTS"])
    hit_next, hit_runs = tables["HIT_NEXT"], tables["HIT_RUNS"]
    branch_p = tables["BRANCH_P"]
    any_bbe = bool((ev_code == EV_BBE).any())

    for t in range(n_ev):
        # Scalar loop top: `if outs == 3: outs = 0; bases = empty`.
        reset = outs == 3
        outs[reset] = 0
        bases[reset] = 0

        ev = ev_seq[t]

        # All deterministic transitions (K / walk / steal / pickoff) in one
        # gather; the EV_BBE row is identity/zero and overridden below.
        # Steal/pickoff on empty bases are identity no-ops in the tables,
        # matching the scalar `if any(bases)` guard.
        runs += det_runs[ev, bases]
        outs += det_outs[ev, bases]
        bases = det_next[ev, bases]

        if any_bbe:
            m = ev == EV_BBE
            k = int(m.sum())
            if k:
                row = cdf[order[t][m]]                      # (k, 4)
                u = rng.random(k)
                cls = (u[:, None] >= row).sum(axis=1)       # 0=out, 1-4=bases
                b = bases[m]

                hit = cls > 0
                branch = (rng.random(k) < branch_p[b, cls]).astype(np.int8)
                nxt = hit_next[branch, b, cls]
                scored = hit_runs[branch, b, cls]

                outs_m = outs[m]
                outs_m[~hit] += 1
                outs[m] = outs_m
                bases[m] = np.where(hit, nxt, b)
                runs[m] += np.where(hit, scored, 0).astype(np.int32)

    return runs


def simulate_games_by_inning_vectorized(ev_code, cdf, innings, tables,
                                        num_simulations, n_innings, rng):
    """Vectorized equivalent of num_simulations calls to simulate_game_by_inning().

    Events run in FIXED chronological order (no permutation). No out counter:
    bases reset only at inning boundaries; strikeouts are no-ops; a pickoff
    removes the lead runner without recording an out — all mirroring the
    scalar simulate_game_by_inning.

    Args:
        ev_code, cdf: from translate_outcomes() (misses already dropped).
        innings: int array (n_events,) — 1-based inning per event, same order.
        n_innings (int): number of inning buckets.

    Returns:
        np.ndarray (num_simulations, n_innings) float64 — CUMULATIVE runs
        through each inning, exactly like simulate_game_by_inning.
    """
    runs_by_inning = np.zeros((num_simulations, n_innings), dtype=np.float64)
    n_ev = len(ev_code)
    if n_ev == 0:
        return runs_by_inning

    if innings.min() < 1 or innings.max() > n_innings:
        raise IndexError(
            f"simulate_games_by_inning_vectorized: inning outside 1..{n_innings}"
        )

    bases = np.zeros(num_simulations, dtype=np.int8)
    prev_inning = None

    walk_next, walk_runs = tables["WALK_NEXT"], tables["WALK_RUNS"]
    steal_next, steal_runs = tables["STEAL_NEXT"], tables["STEAL_RUNS"]
    pick_next = tables["PICK_NEXT"]
    hit_next, hit_runs = tables["HIT_NEXT"], tables["HIT_RUNS"]
    branch_p = tables["BRANCH_P"]

    for t in range(n_ev):
        inning = int(innings[t])
        if inning != prev_inning:
            bases[:] = 0                    # new half-inning for this team
            prev_inning = inning
        i = inning - 1
        code = int(ev_code[t])

        if code == EV_STRIKEOUT:
            continue                        # out: no runs, bases unchanged
        elif code == EV_WALK:
            runs_by_inning[:, i] += walk_runs[bases]
            bases = walk_next[bases]
        elif code == EV_STEAL:
            runs_by_inning[:, i] += steal_runs[bases]
            bases = steal_next[bases]
        elif code == EV_PICKOFF:
            bases = pick_next[bases]        # lead runner removed, no out
        else:                               # EV_BBE
            u = rng.random(num_simulations)
            cls = (u[:, None] >= cdf[t][None, :]).sum(axis=1)
            hit = cls > 0
            branch = (rng.random(num_simulations) < branch_p[bases, cls]).astype(np.int8)
            nxt = hit_next[branch, bases, cls]
            scored = hit_runs[branch, bases, cls]
            runs_by_inning[:, i] += np.where(hit, scored, 0)
            bases = np.where(hit, nxt, bases).astype(np.int8)

    return np.cumsum(runs_by_inning, axis=1)
