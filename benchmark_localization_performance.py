# Fault localization batch processing.
#
# ARCHITECTURE CHANGE — position-level parallelism
# ─────────────────────────────────────────────────
# Previous design: one worker per seed.
#   Problem: with N seeds having work, only N cores are ever busy.
#   With 70 seeds total and only 8-9 having remaining positions, 55+
#   cores sat completely idle the entire run.
#
# New design: one task per (seed, link, ratio) position.
#   Every position is submitted as an independent apply_async task.
#   The pool's 64 workers pull from a flat task list and stay busy
#   until the very last position is done — no core goes idle early
#   because one seed finished before others.
#
#   Cost: workers can no longer share reference data / FaultLocalizer
#   across positions for the same seed "for free". Recovered via a
#   worker-local cache (_worker_seed_cache): the first time a worker
#   sees a seed it loads reference data and builds FaultLocalizer once,
#   then reuses them for all subsequent positions of that seed that
#   happen to land on the same worker. This is best-effort reuse —
#   correctness does not depend on it; the worst case is a re-load,
#   which is what the old per-position path did anyway.
#
# All prior fixes retained:
#   FIX 1 — float rounding resume bug: _ratio_key() integer keys.
#   FIX 2 — queue timeout to detect silent worker death.
#   FIX 3 — Welford online stats instead of growing lists.

import gc
import json
import multiprocessing as mp
import os
import time
from multiprocessing import Pool, Queue

import numpy as np
from scipy.signal import hilbert, savgol_filter
from tqdm import tqdm

from complex_network.detection.naive_localization_correlation import FaultLocalizer
from complex_network.networks.network_factory import generate_network
from complex_network.networks.network_path_search import _spec_fingerprint
from complex_network.networks.network_spec import NetworkSpec

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
OUTPUT_DIR       = '/home/baruva/network_cache/olcr_data_ganesh'
OUTPUT_STATS_DIR = '/home/baruva/network_cache/olcr_data_ganesh/performance_stats_correlation'
NI               = 8
NE               = 1
NUM_WORKERS      = 64
COHERENCE_LENGTH = 1.1e-6        # optical path length, metres
SPATIAL_RES      = COHERENCE_LENGTH / 3  # physical distance, metres

# How many positions a worker processes between cache clears.
CACHE_CLEAR_INTERVAL = 20

# Bounded result queue — workers block when full (backpressure).
MAX_QUEUE_SIZE = NUM_WORKERS * 4

# How long the main loop waits on queue.get() before checking for
# crashed workers. 5 minutes is generous; lower if positions are fast.
QUEUE_TIMEOUT_S = 300

# ------------------------------------------------------------------ #
# Worker-local state (module globals, set at fork time by _init_worker)
# ------------------------------------------------------------------ #
_shared_queue:      Queue = None  # type: ignore
_worker_seed_cache: dict  = {}    # seed -> built state dict
_worker_pos_count:  int   = 0     # for periodic GC trigger


def _init_worker(q: Queue):
    """Pool initializer: runs once per worker at fork time."""
    global _shared_queue, _worker_seed_cache, _worker_pos_count
    _shared_queue      = q
    _worker_seed_cache = {}
    _worker_pos_count  = 0


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _ratio_key(ratio: float) -> int:
    """FIX 1: integer key avoids IEEE-754 float comparison bugs."""
    return int(round(float(ratio) * 100))


def _compute_spatial_error(network, *, true_link_index, true_ratio,
                            pred_link_tuple, pred_ratio):
    true_pos = network.spatial_position_within_link(
        link_index=true_link_index, fractional_ratio=true_ratio)
    if pred_link_tuple is None or pred_ratio is None:
        return None
    try:
        pred_pos = network.spatial_position_within_link(
            node_tuple=tuple(pred_link_tuple),
            fractional_ratio=float(pred_ratio))
        return float(np.linalg.norm(pred_pos - true_pos))
    except Exception:
        return None


def _envelope(signal, smooth_window):
    """DC-remove -> Hilbert envelope -> Savitzky-Golay smooth."""
    s   = signal - np.mean(signal)
    env = np.abs(hilbert(s))
    del s
    return savgol_filter(env, window_length=smooth_window, polyorder=3)


def _smooth_window(opl):
    dx = float(np.abs(opl[1] - opl[0]))
    w  = int(np.ceil(COHERENCE_LENGTH / dx))
    return w + 1 if w % 2 == 0 else w


def _clear_localizer_caches(localizer):
    localizer._path_validation_cache.clear()
    localizer._reduced_variants_cache.clear()
    localizer._simple_path_cache.clear()


def _build_seed_state(seed: int) -> dict:
    """
    Build network, reference envelopes, and FaultLocalizer for a seed.
    Called at most once per seed per worker process; result is cached
    in _worker_seed_cache.
    """
    spec = NetworkSpec(
        num_external_nodes=NE,
        num_internal_nodes=NI,
        network_type='delaunay',
        network_shape='slab',
        network_size=(200e-6, 200e-6),
        external_offset=10e-6,
        random_seed=seed,
        fully_connected=True,
        node_S_mat_type='neumann',
    )
    network     = generate_network(spec)
    fingerprint = _spec_fingerprint(spec)
    sources     = [node.index for node in network.external_nodes]

    opl    = np.load(os.path.join(OUTPUT_DIR, f'{fingerprint}_opls.npy'))
    ref_s1 = np.load(os.path.join(OUTPUT_DIR, f'{fingerprint}_ref_s1.npy'))
    ref_s2 = np.load(os.path.join(OUTPUT_DIR, f'{fingerprint}_ref_s2.npy'))

    sw         = _smooth_window(opl)
    env_ref_s1 = _envelope(ref_s1, sw)
    env_ref_s2 = _envelope(ref_s2, sw)
    del ref_s1, ref_s2

    olcr_ref_dict = {
        sources[0]: (opl, env_ref_s1),
        sources[1]: (opl, env_ref_s2),
    }

    localizer = FaultLocalizer(
        network=network,
        source_indices=sources,
        max_hops=14,
        n_index=1.5,
        coherence_length=3e-9,
        use_multiprocessing=False,
        opls=opl,
        aggregation_method='sum',
    )

    return {
        'network':     network,
        'localizer':   localizer,
        'olcr_ref':    olcr_ref_dict,
        'opl':         opl,
        'sw':          sw,
        'sources':     sources,
        'fingerprint': fingerprint,
    }


# ------------------------------------------------------------------ #
# Per-position worker function
# ------------------------------------------------------------------ #

def process_position(task: tuple):
    """
    Process a single (seed, link_index, ratio) position.

    Seed state is cached per worker process in _worker_seed_cache so
    that network build + reference envelope + FaultLocalizer are only
    paid once per seed per worker, not once per position.
    """
    global _worker_pos_count

    seed, link_index, ratio = task
    result = None

    try:
        # ── Seed state: load once, reuse for subsequent positions ────────
        if seed not in _worker_seed_cache:
            _worker_seed_cache[seed] = _build_seed_state(seed)
        state = _worker_seed_cache[seed]

        network     = state['network']
        localizer   = state['localizer']
        olcr_ref    = state['olcr_ref']
        opl         = state['opl']
        sw          = state['sw']
        sources     = state['sources']
        fingerprint = state['fingerprint']

        # ── Load and process perturbed data ──────────────────────────────
        pert_s1 = np.load(os.path.join(
            OUTPUT_DIR,
            f'{fingerprint}_pert_link{link_index}_r{ratio:.2f}_s1.npy'))
        pert_s2 = np.load(os.path.join(
            OUTPUT_DIR,
            f'{fingerprint}_pert_link{link_index}_r{ratio:.2f}_s2.npy'))

        env_pert_s1 = _envelope(pert_s1, sw)
        env_pert_s2 = _envelope(pert_s2, sw)
        del pert_s1, pert_s2

        olcr_pert_dict = {
            sources[0]: (opl, env_pert_s1),
            sources[1]: (opl, env_pert_s2),
        }

        best_link, best_position, score = localizer.localize_fault(
            olcr_ref, olcr_pert_dict)

        del env_pert_s1, env_pert_s2, olcr_pert_dict

        # ── Score ────────────────────────────────────────────────────────
        true_link = (
            network.get_link(link_index).sorted_connected_nodes[0],
            network.get_link(link_index).sorted_connected_nodes[1],
        )
        true_link_length = network.get_link(link_index).length
        threshold        = SPATIAL_RES / true_link_length

        spatial_error = _compute_spatial_error(
            network,
            true_link_index=link_index,
            true_ratio=ratio,
            pred_link_tuple=best_link,
            pred_ratio=best_position,
        )

        if best_link == true_link:
            position_error = abs(best_position - ratio)
            category = ('correct' if position_error <= threshold
                        else 'correct_link_wrong_position')
        else:
            position_error = None
            category = 'wrong_link'

        result = {
            'seed':           seed,
            'link_index':     link_index,
            'ratio':          ratio,
            'true_link':      true_link,
            'best_link':      best_link,
            'best_position':  best_position,
            'score':          score,
            'position_error': position_error,
            'spatial_error':  spatial_error,
            'threshold':      threshold,
            'category':       category,
        }

    except Exception as e:
        import traceback
        print(f'\n[seed={seed}] Error at link={link_index}, '
              f'ratio={ratio:.2f}: {e}')
        traceback.print_exc()

    # ── Periodic localizer cache clear ───────────────────────────────
    _worker_pos_count += 1
    if _worker_pos_count % CACHE_CLEAR_INTERVAL == 0:
        for st in _worker_seed_cache.values():
            _clear_localizer_caches(st['localizer'])
        gc.collect()

    _shared_queue.put(result)   # blocks if queue full (backpressure)


# ------------------------------------------------------------------ #
# Resume
# ------------------------------------------------------------------ #

def load_completed_positions(seed_indices):
    """Return {seed: set of (link_index, ratio_int)} already on disk."""
    completed = {}
    for seed in seed_indices:
        done = set()
        path = os.path.join(OUTPUT_STATS_DIR,
                            f'results_ni{NI}_seed_delta{seed}.jsonl')
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        li  = int(rec['true_link_index'])
                        if 'true_position_int' in rec:
                            ri = int(rec['true_position_int'])
                        else:
                            ri = _ratio_key(rec['true_position'])
                        done.add((li, ri))
                    except Exception:
                        pass
        completed[seed] = done
    return completed


# ------------------------------------------------------------------ #
# Task collection — flat list of (seed, link_index, ratio)
# ------------------------------------------------------------------ #

def collect_all_tasks(seed_indices, completed_per_seed):
    """
    Returns (tasks, skipped_count) where tasks is a flat list of
    (seed, link_index, ratio) for every remaining position whose
    perturbed files exist on disk.
    """
    tasks         = []
    skipped_total = 0

    for seed in seed_indices:
        spec = NetworkSpec(
            num_external_nodes=NE,
            num_internal_nodes=NI,
            network_type='delaunay',
            network_shape='slab',
            network_size=(200e-6, 200e-6),
            external_offset=10e-6,
            random_seed=seed,
            fully_connected=True,
            node_S_mat_type='neumann',
        )
        network     = generate_network(spec)
        fingerprint = _spec_fingerprint(spec)
        done        = completed_per_seed.get(seed, set())

        for link in network.internal_links:
            li          = link.index
            link_length = network.get_link(li).length
            ratios      = np.unique(np.round(
                np.arange(5e-6, link_length, 5e-6) / link_length, 2))
            ratios      = ratios[(ratios > 0) & (ratios < 1)]

            for ratio in ratios:
                key = (li, _ratio_key(ratio))
                if key in done:
                    skipped_total += 1
                    continue
                p1 = os.path.join(
                    OUTPUT_DIR,
                    f'{fingerprint}_pert_link{li}_r{ratio:.2f}_s1.npy')
                p2 = os.path.join(
                    OUTPUT_DIR,
                    f'{fingerprint}_pert_link{li}_r{ratio:.2f}_s2.npy')
                if os.path.exists(p1) and os.path.exists(p2):
                    tasks.append((seed, li, float(ratio)))

    return tasks, skipped_total


# ------------------------------------------------------------------ #
# Disk save
# ------------------------------------------------------------------ #

def save_individual_result(result):
    if result is None:
        return
    record = {
        'true_link_index':      result['link_index'],
        'true_link_tuple':      list(result['true_link']),
        'true_position':        float(result['ratio']),
        'true_position_int':    _ratio_key(result['ratio']),
        'predicted_link_tuple': list(result['best_link']) if result['best_link'] else None,
        'predicted_position':   _safe_float(result['best_position']),
        'score':                _safe_float(result['score']),
        'position_error':       _safe_float(result['position_error']),
        'spatial_error':        _safe_float(result['spatial_error']),
        'threshold':            float(result['threshold']),
        'category':             result['category'],
    }
    path = os.path.join(OUTPUT_STATS_DIR,
                        f'results_ni{NI}_seed_delta{result["seed"]}.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ------------------------------------------------------------------ #
# Welford online mean/variance — O(1) RAM per tracked stat
# ------------------------------------------------------------------ #

class _Welford:
    __slots__ = ('n', 'mean', '_M2')

    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self._M2  = 0.0

    def update(self, value):
        if value is None or not np.isfinite(value):
            return
        self.n   += 1
        delta     = value - self.mean
        self.mean += delta / self.n
        self._M2  += delta * (value - self.mean)

    @property
    def std(self):
        return float(np.sqrt(self._M2 / (self.n - 1))) if self.n > 1 else 0.0


# ------------------------------------------------------------------ #
# In-memory aggregation
# ------------------------------------------------------------------ #

def _init_seed_stats(seed):
    return {
        'seed':                        seed,
        'ni':                          NI,
        'correct_predictions':         0,
        'correct_link_wrong_position': 0,
        'wrong_link':                  0,
        'total_predictions':           0,
        '_pos_err_wf':                 _Welford(),
        '_spa_err_wf':                 _Welford(),
        'per_link_results':            {},
        'per_link_accuracy':           {},
    }


def update_seed_stats(seed_stats, result):
    if result is None:
        return
    seed     = result['seed']
    li       = result['link_index']
    category = result['category']
    stats    = seed_stats[seed]

    stats['total_predictions'] += 1

    if li not in stats['per_link_results']:
        stats['per_link_results'][li] = {
            'correct':                0,
            'correct_link_wrong_pos': 0,
            'wrong_link':             0,
            '_pos_err_wf':            _Welford(),
            '_spa_err_wf':            _Welford(),
        }
    lr = stats['per_link_results'][li]

    if category == 'correct':
        stats['correct_predictions'] += 1
        lr['correct']                += 1
        stats['_pos_err_wf'].update(result['position_error'])
        lr['_pos_err_wf'].update(result['position_error'])
    elif category == 'correct_link_wrong_position':
        stats['correct_link_wrong_position'] += 1
        lr['correct_link_wrong_pos']         += 1
        stats['_pos_err_wf'].update(result['position_error'])
        lr['_pos_err_wf'].update(result['position_error'])
    else:
        stats['wrong_link'] += 1
        lr['wrong_link']    += 1

    stats['_spa_err_wf'].update(result['spatial_error'])
    lr['_spa_err_wf'].update(result['spatial_error'])


def finalise_per_link_accuracy(seed_stats):
    for stats in seed_stats.values():
        for li, ld in stats['per_link_results'].items():
            total = (ld['correct'] + ld['correct_link_wrong_pos']
                     + ld['wrong_link'])
            if total == 0:
                continue
            stats['per_link_accuracy'][li] = {
                'accuracy':            ld['correct'] / total * 100,
                'correct':             ld['correct'],
                'total':               total,
                'mean_position_error': ld['_pos_err_wf'].mean,
                'std_position_error':  ld['_pos_err_wf'].std,
                'mean_spatial_error':  ld['_spa_err_wf'].mean,
                'std_spatial_error':   ld['_spa_err_wf'].std,
            }


# ------------------------------------------------------------------ #
# Final JSON save
# ------------------------------------------------------------------ #

def save_seed_statistics(seed_stats):
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)
    for seed, stats in seed_stats.items():
        path = os.path.join(
            OUTPUT_STATS_DIR,
            f'performance_stats_correlation_ni{NI}_seed_delta{seed}.json')
        obj = {
            'seed':                        int(stats['seed']),
            'ni':                          int(stats['ni']),
            'correct_predictions':         int(stats['correct_predictions']),
            'correct_link_wrong_position': int(stats['correct_link_wrong_position']),
            'wrong_link':                  int(stats['wrong_link']),
            'total_predictions':           int(stats['total_predictions']),
            'position_error_mean':         stats['_pos_err_wf'].mean,
            'position_error_std':          stats['_pos_err_wf'].std,
            'position_error_n':            stats['_pos_err_wf'].n,
            'spatial_error_mean':          stats['_spa_err_wf'].mean,
            'spatial_error_std':           stats['_spa_err_wf'].std,
            'spatial_error_n':             stats['_spa_err_wf'].n,
            'per_link_results': {
                str(li): {
                    'correct':                int(ld['correct']),
                    'correct_link_wrong_pos': int(ld['correct_link_wrong_pos']),
                    'wrong_link':             int(ld['wrong_link']),
                    'position_error_mean':    ld['_pos_err_wf'].mean,
                    'position_error_std':     ld['_pos_err_wf'].std,
                    'spatial_error_mean':     ld['_spa_err_wf'].mean,
                    'spatial_error_std':      ld['_spa_err_wf'].std,
                }
                for li, ld in stats['per_link_results'].items()
            },
            'per_link_accuracy': {
                str(li): {
                    'accuracy':            float(la['accuracy']),
                    'correct':             int(la['correct']),
                    'total':               int(la['total']),
                    'mean_position_error': float(la['mean_position_error']),
                    'std_position_error':  float(la['std_position_error']),
                    'mean_spatial_error':  float(la['mean_spatial_error']),
                    'std_spatial_error':   float(la['std_spatial_error']),
                }
                for li, la in stats['per_link_accuracy'].items()
            },
        }
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

        if stats['total_predictions'] > 0:
            acc    = stats['correct_predictions'] / stats['total_predictions'] * 100
            se_wf  = stats['_spa_err_wf']
            se_str = f"{se_wf.mean:.2e} m" if se_wf.n > 0 else "N/A"
            print(f"  Seed {seed:4d}: "
                  f"{stats['correct_predictions']:>5}/{stats['total_predictions']:<5} "
                  f"correct ({acc:5.1f}%)  |  mean spatial err: {se_str}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

    # seed_indices = list(range(0,9, 1))
    seed_indices = [66090]
    if not seed_indices:
        raise ValueError("seed_indices is empty.")

    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"FAULT LOCALIZATION BATCH — POSITION-LEVEL PARALLELISM")
    print(f"{'='*80}")
    print(f"  Seeds:          {seed_indices[0]} to {seed_indices[-1]} "
          f"({len(seed_indices)} total)")
    print(f"  Internal nodes: {NI}   External nodes: {NE}")
    print(f"  Workers:        {NUM_WORKERS}")
    print(f"  Queue max size: {MAX_QUEUE_SIZE}")
    print(f"  Queue timeout:  {QUEUE_TIMEOUT_S}s")
    print(f"{'='*80}\n")

    total_start = time.time()

    # ── 1. Resume ────────────────────────────────────────────────────
    print("Scanning existing JSONL files for completed positions...")
    t0                 = time.time()
    completed_per_seed = load_completed_positions(seed_indices)
    already_done       = sum(len(v) for v in completed_per_seed.values())
    print(f"  Found {already_done} already-completed positions "
          f"[{time.time()-t0:.1f}s]")

    # ── 2. Flat task list ─────────────────────────────────────────────
    print("Collecting remaining tasks...")
    t0 = time.time()
    all_tasks, skipped = collect_all_tasks(seed_indices, completed_per_seed)
    total_tasks        = len(all_tasks)
    active_seeds       = len({t[0] for t in all_tasks})
    print(f"  {total_tasks} positions remaining across {active_seeds} seeds  "
          f"({skipped} skipped as already done)  "
          f"[{time.time()-t0:.1f}s]\n")

    if total_tasks == 0:
        print("Nothing left to compute. Exiting.")
        raise SystemExit(0)

    # ── 3. In-memory aggregation ──────────────────────────────────────
    seed_stats = {s: _init_seed_stats(s) for s in seed_indices}

    # ── 4. Queue created before pool (inherited at fork) ─────────────
    queue = Queue(maxsize=MAX_QUEUE_SIZE)

    completed_pos = 0
    failed_pos    = 0

    print(f"Starting {NUM_WORKERS} worker processes "
          f"({total_tasks} tasks dispatched)...")
    t0 = time.time()

    with Pool(processes=NUM_WORKERS,
              initializer=_init_worker,
              initargs=(queue,)) as pool:

        handles = [
            pool.apply_async(process_position, args=(task,))
            for task in all_tasks
        ]

        with tqdm(total=total_tasks,
                  desc="Positions processed",
                  unit="pos",
                  dynamic_ncols=True) as pbar:

            received = 0
            while received < total_tasks:
                # FIX 2: timeout so silent worker death doesn't hang main.
                try:
                    item = queue.get(timeout=QUEUE_TIMEOUT_S)
                except Exception:
                    worker_crashed = False
                    for h in handles:
                        try:
                            h.get(timeout=0)
                        except mp.TimeoutError:
                            pass
                        except Exception as worker_exc:
                            print(f'\n[FATAL] Worker exception: {worker_exc}')
                            worker_crashed = True
                    if worker_crashed:
                        raise RuntimeError(
                            "One or more workers crashed. See stderr.")
                    continue

                received += 1
                save_individual_result(item)
                update_seed_stats(seed_stats, item)

                if item is None:
                    failed_pos += 1
                else:
                    completed_pos += 1

                pbar.update(1)
                pbar.set_postfix(failed=failed_pos, refresh=False)

        for h in handles:
            h.get()

    elapsed = time.time() - t0
    print(f"\nProcessing complete: {elapsed:.1f}s  "
          f"({total_tasks / elapsed:.1f} pos/s)\n")

    # ── 5. Finalise and save ──────────────────────────────────────────
    print("Computing per-link accuracy and saving statistics...")
    finalise_per_link_accuracy(seed_stats)
    save_seed_statistics(seed_stats)

    # ── 6. Summary ────────────────────────────────────────────────────
    all_correct = sum(s['correct_predictions'] for s in seed_stats.values())
    all_total   = sum(s['total_predictions']   for s in seed_stats.values())
    overall_acc = (all_correct / all_total * 100) if all_total > 0 else 0
    wall        = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"ALL DONE")
    print(f"{'='*80}")
    print(f"  Skipped (already done):  {already_done}")
    print(f"  Positions completed:     {completed_pos}/{total_tasks}")
    print(f"  Positions failed:        {failed_pos}/{total_tasks}")
    print(f"  Overall accuracy:        {overall_acc:.1f}%  "
          f"({all_correct}/{all_total})")
    print(f"  Total wall time:         {wall:.1f}s  ({wall/60:.1f} min)")
    print(f"{'='*80}")
