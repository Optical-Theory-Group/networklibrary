"""
DORT fault-localization performance study -- SEGMENT (index-perturbation) faults.

This benchmark is the exact single-shot DORT algorithm (complex_network.detection.dort.DORT),
run as a large statistical sweep. The localizer is NOT re-implemented here: every fault is
localised by calling DORT.localise_fault, so the benchmark and the single-shot code can never
drift apart. The only thing this file adds is the sweep harness (seeds x faults) and the
efficiency structure that a millions-of-faults study needs.

The only difference from benchmark_dort_node.py is the fault model: instead of a point
partial-reflector node, each fault is a finite index-perturbation SEGMENT of physical length
FAULT_SIZE (um) and index step FAULT_INDEX_CHANGE, centred at the fractional position x_f.

Efficiency structure
--------------------
The expensive part of DORT is the per-network precompute (the healthy-network field solve over
all frequencies), which depends only on the healthy network, not on the fault. So we build ONE
DORT object per seed in the parent process and share it read-only with a fork pool via
copy-on-write. Each worker then only does the cheap per-fault work: build the faulty S-matrix
and call localise_fault. The precompute is paid once per seed, never per fault.

Pipeline per fault (all blind -- no oracle fault position is ever used):

  1. COARSE LINK ID.  DORT ranks every link by projecting the chosen per-frequency signature
     (the dominant time-reversal eigenvector v1 for method='eigen', or the strongest DeltaS
     column for method='best_column') onto each link's coarse Green's basis, weighted by
     w(k) = S(k)*purity(k),  purity = 1 - sqrt(lam2/lam1). pred_link = top-ranked link.

  2. POSITION LOCALIZATION on pred_link (whatever coarse ID picked -- coupled, as a real
     deployment runs), returning both estimates from a single call (sum='both'):
       Incoherent  P_inc(x) = sum_k w(k) |<sig, g_hat(x,k)>|^2      (phase-free)
       Coherent    P_coh(x) = | sum_k w(k) <sig, g_hat(x,k)> / ||g(x,k)|| |
     PHASE_ANCHORING toggles DORT's blind coherent phase anchor (default off).

  3. ERROR METRIC: physical Euclidean distance (um) between the estimated point and the true
     fault point, via net.spatial_position_within_link. This stays well-defined even when
     pred_link != true link. Same-link fractional errors are also recorded, but only when the
     coarse ID picked the true link.

Sweeps NI in {10, 30, 50, 80, 100}. Results saved per (ni, seed) as atomic JSON.

Run:   nohup python benchmark_dort_segment.py > dort_segment_study.log 2>&1 &
Smoke: python benchmark_dort_segment.py --smoke
"""
import os, sys, time, json, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import multiprocessing as mp

from complex_network.networks.network_factory import generate_network
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.detection.dort import DORT

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
CENTRAL_WAVELENGTH = 1000e-9
BANDWIDTH = 400e-9
K_MIN = 2 * np.pi / (CENTRAL_WAVELENGTH + BANDWIDTH / 2)
K_MAX = 2 * np.pi / (CENTRAL_WAVELENGTH - BANDWIDTH / 2)
K_CTR = 2 * np.pi / CENTRAL_WAVELENGTH

NI_VALUES = [10, 30, 50, 80, 100]   # internal-node counts to sweep
N_EXTERNAL = 1                      # NE
NUM_SEEDS = 100
INIT_SEED = 0
NUM_FAULTS_PER_SEED = int(1e3)
N_FREQ = 500
N_WORKERS = 60
N_X = 2000                          # fine position grid (DORT fine_grid_sampling)
NXC = 48                            # coarse grid for the link-ID basis (DORT coarse_grid_sampling)
EDGE_MARGIN = 0.05                  # exclude thin edge layer from peak-finding

# --- segment fault model ---
FAULT_SIZE = 0.2                   # physical segment length [um]
FAULT_INDEX_CHANGE = 0.001         # refractive-index step dn of the segment

# --- single-shot DORT options (exactly the knobs of DORT.localise_fault) ---
METHOD = 'eigen'                   # 'eigen' (default) or 'best_column'
PHASE_ANCHORING = False            # blind coherent phase anchor (DORT default is False)

RELIABLE_TOL_UM = 1.0              # physical-distance threshold for "reliable" (~ central wavelength)

STUDY_TAG = f'dort_segment_s{FAULT_SIZE}_dn{FAULT_INDEX_CHANGE}_{METHOD}'
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_output')


def make_band(n_freq):
    """Gaussian-windowed k grid centred on the band -> (k_vals, weights S(k))."""
    k_vals = np.linspace(K_MIN, K_MAX, n_freq)
    sigma_k = (k_vals[-1] - k_vals[0]) / 4.0
    S = np.exp(-0.5 * ((k_vals - K_CTR) / sigma_k) ** 2)
    S /= np.sum(S)
    return k_vals, S


def make_network(seed, ni=10, ne=1):
    spec = NetworkSpec(
        num_external_nodes=ne, num_internal_nodes=ni,
        network_type='delaunay', network_shape='slab',
        network_size=(200e-6, 200e-6), external_offset=10e-6,
        random_seed=seed, fully_connected=True,
        node_S_mat_type='neumann')
    return generate_network(spec)


def init_worker():
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_v] = '1'


# ---------------------------------------------------------------------
# per-seed DORT object (built in the parent, inherited by workers via fork COW)
# ---------------------------------------------------------------------
def build_dort(seed, ni, n_freq):
    """Build one healthy network and its DORT precompute. Runs once per seed."""
    net = make_network(seed, ni=ni, ne=N_EXTERNAL)
    k_vals, S = make_band(n_freq)
    # healthy S(k) over the same (ascending) k grid handed to DORT as the spectrum.
    healthy_S = np.array([net.get_S_ee(k0) for k0 in k_vals])
    dort = DORT(
        healthy_S_matrix_array=healthy_S,
        spectrum=(k_vals, S),
        healthy_network=net,
        coarse_grid_sampling=NXC,
        fine_grid_sampling=N_X,
        edge_margin=EDGE_MARGIN,
    )
    dort._seed = seed
    return dort


_CURRENT_DORT = None   # set before Pool creation; inherited via fork COW


# ---------------------------------------------------------------------
# per-fault worker
# ---------------------------------------------------------------------
def process_fault(task):
    """task = (fault_idx, link_idx, true_frac_pos). Reads the current seed's DORT
    object from the module global inherited via fork."""
    dort = _CURRENT_DORT
    net = dort.healthy_network
    fault_idx, link_idx, x_f = task

    try:
        # ---- place a finite index-perturbation SEGMENT -> faulty S(k) ----
        # half-width as a fraction of this link's physical length (FAULT_SIZE is in um).
        link_length_um = float(dort.link_lengths[link_idx] * 1e6)
        half_frac = (FAULT_SIZE / 2) / link_length_um
        pb = NetworkPerturbator(net)
        pb.add_index_perturbation_segment(
            link_index=link_idx,
            size=(x_f - half_frac, x_f + half_frac),
            dn_value=FAULT_INDEX_CHANGE)
        faulty_S = np.array([pb.perturbed_network.get_S_ee(k0) for k0 in dort.kvalues])
        del pb

        # ---- single-shot DORT localisation (coarse link ID + fine position, both estimates) ----
        pred_link, coh_est_frac, inc_est_frac = dort.localise_fault(
            method=METHOD, sum='both', phase_anchoring=PHASE_ANCHORING,
            faulty_S_matrix_array=faulty_S)

        link_rank_order = dort.link_rank_order
        link_id_rank = int(np.where(link_rank_order == link_idx)[0][0])
        link_id_correct = (pred_link == link_idx)
        purity_mean = float(dort.purity.mean())

        # ---- physical-distance error (well-defined even when pred_link != link_idx) ----
        pos_true = net.spatial_position_within_link(link_index=link_idx, fractional_ratio=x_f)
        pos_inc = net.spatial_position_within_link(link_index=pred_link, fractional_ratio=inc_est_frac)
        pos_coh = net.spatial_position_within_link(link_index=pred_link, fractional_ratio=coh_est_frac)
        inc_err_um = float(np.linalg.norm(pos_inc - pos_true)) * 1e6
        coh_err_um = float(np.linalg.norm(pos_coh - pos_true)) * 1e6

        # same-link fractional error only meaningful when coarse ID picked the true link
        inc_frac_err = abs(inc_est_frac - x_f) if link_id_correct else None
        coh_frac_err = abs(coh_est_frac - x_f) if link_id_correct else None

        return {
            'seed': dort._seed, 'fault_idx': fault_idx, 'link_idx': link_idx,
            'link_length_um': link_length_um,
            'true_frac_pos': x_f, 'n_freq': N_FREQ,
            'purity_mean': purity_mean,
            'pred_link': int(pred_link), 'link_id_rank': link_id_rank,
            'link_id_correct': bool(link_id_correct),
            'inc_est_frac': float(inc_est_frac), 'inc_err_um': inc_err_um,
            'inc_frac_err': inc_frac_err,
            'inc_reliable': bool(inc_err_um < RELIABLE_TOL_UM),
            'coh_est_frac': float(coh_est_frac), 'coh_err_um': coh_err_um,
            'coh_frac_err': coh_frac_err,
            'coh_reliable': bool(coh_err_um < RELIABLE_TOL_UM),
        }
    except Exception as e:
        return {'seed': getattr(dort, '_seed', None), 'fault_idx': fault_idx,
                'error': str(e), 'traceback': traceback.format_exc()}


# ---------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------
def save_results_atomic(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def make_tasks(seed, n_links):
    """Balanced link sampling (equal count per link, shuffled) + uniform pos."""
    rng = np.random.default_rng(seed)
    link_seq = np.tile(np.arange(n_links),
                       NUM_FAULTS_PER_SEED // n_links + 1)[:NUM_FAULTS_PER_SEED]
    rng.shuffle(link_seq)
    pos = rng.uniform(EDGE_MARGIN, 1.0 - EDGE_MARGIN, NUM_FAULTS_PER_SEED)
    return [(i, int(link_seq[i]), float(pos[i])) for i in range(NUM_FAULTS_PER_SEED)]


def run_seed(ni, seed, out_dir, n_faults, n_workers):
    global _CURRENT_DORT
    out_path = f'{out_dir}/R_{STUDY_TAG}_ni{ni}_seed{seed}.json'
    if os.path.exists(out_path):
        print(f"  seed {seed}: SKIP (already done)")
        return

    t0 = time.time()
    print(f"  seed {seed}: building ni={ni} network + DORT precompute...")
    t_pre = time.time()
    dort = build_dort(seed, ni, N_FREQ)
    dt = time.time() - t_pre
    n_links = int(dort.link_lengths.shape[0])
    length_um = dort.link_lengths * 1e6
    print(f"    precompute {dt:.0f}s, {n_links} links, "
          f"L in [{length_um.min():.1f}, {length_um.max():.1f}] um")

    tasks = make_tasks(seed, n_links)[:n_faults]
    _CURRENT_DORT = dort
    t_pool = time.time()
    chunksize = max(1, len(tasks) // (n_workers * 4))
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=init_worker) as pool:
        results = pool.map(process_fault, tasks, chunksize=chunksize)
    _CURRENT_DORT = None
    dt_pool = time.time() - t_pool

    n_err = sum(1 for r in results if 'error' in r)
    ok = [r for r in results if 'error' not in r]
    link_acc = np.mean([r['link_id_correct'] for r in ok]) if ok else float('nan')
    inc_med = np.median([r['inc_err_um'] for r in ok]) if ok else float('nan')
    coh_med = np.median([r['coh_err_um'] for r in ok]) if ok else float('nan')
    print(f"    {len(results)} faults in {dt_pool:.0f}s ({len(results)/dt_pool:.1f}/s), "
          f"{n_err} errors | link-ID acc={link_acc:.2f} "
          f"inc_err_med={inc_med:.3f}um coh_err_med={coh_med:.3f}um")

    save_results_atomic(out_path, {
        'ni': ni, 'seed': seed, 'n_links': n_links, 'n_freq': N_FREQ,
        'config': {'NE': N_EXTERNAL, 'fault_size_um': FAULT_SIZE,
                   'fault_index_change': FAULT_INDEX_CHANGE,
                   'n_x': N_X, 'nxc': NXC, 'edge_margin': EDGE_MARGIN,
                   'method': METHOD, 'phase_anchoring': PHASE_ANCHORING},
        'rows': results,
    })
    print(f"    saved {out_path} ({time.time()-t0:.0f}s total)")


def main():
    print("=== DORT SEGMENT-FAULT Localization Study ===")
    print(f"NI={NI_VALUES}, NE={N_EXTERNAL}, seeds={NUM_SEEDS} (start={INIT_SEED}), "
          f"faults/seed={NUM_FAULTS_PER_SEED}, N_FREQ={N_FREQ}, workers={N_WORKERS}, "
          f"fault_size={FAULT_SIZE}um dn={FAULT_INDEX_CHANGE}, "
          f"method={METHOD}, phase_anchoring={PHASE_ANCHORING}")
    t_total = time.time()
    for ni in NI_VALUES:
        out_dir = f'{OUT_ROOT}/{STUDY_TAG}_{ni}'
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}\nNI = {ni}\n{'='*60}")
        for seed in range(INIT_SEED, INIT_SEED + NUM_SEEDS):
            run_seed(ni, seed, out_dir, NUM_FAULTS_PER_SEED, N_WORKERS)
    print(f"\nALL DONE in {(time.time()-t_total)/3600:.2f}h")


def smoke():
    """One seed, 20 faults, ni=10: assert the pipeline is well-formed."""
    print("=== SMOKE TEST (1 seed, 20 faults, ni=10) ===")
    out_dir = f'{OUT_ROOT}/{STUDY_TAG}_smoke'
    os.makedirs(out_dir, exist_ok=True)
    run_seed(10, INIT_SEED, out_dir, 20, 4)
    path = f'{out_dir}/R_{STUDY_TAG}_ni10_seed{INIT_SEED}.json'
    payload = json.load(open(path))
    rows = payload['rows']
    errs = [r for r in rows if 'error' in r]
    assert not errs, f"{len(errs)} faults errored: {errs[0].get('traceback','')[:400]}"
    for r in rows:
        assert r['inc_err_um'] >= 0.0 and r['coh_err_um'] >= 0.0, r
        assert 0 <= r['link_id_rank'], r
        if r['link_id_correct']:
            assert r['inc_frac_err'] is not None and r['coh_frac_err'] is not None, r
        else:
            assert r['inc_frac_err'] is None and r['coh_frac_err'] is None, r
    n_links = payload['n_links']
    link_acc = np.mean([r['link_id_correct'] for r in rows])
    print(f"SMOKE OK: {len(rows)} faults, no errors, link-ID top-1 acc={link_acc:.2f} "
          f"(chance={1/n_links:.3f})")
    os.remove(path)


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        smoke()
    else:
        main()
