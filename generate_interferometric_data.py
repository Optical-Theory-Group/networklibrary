#!/usr/bin/env python3

import numpy as np
import os
import time
from complex_network.networks.network_factory import generate_network
from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.interferometry.olcr import OLCR
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_path_search import _spec_fingerprint

# ── Top-level configuration ────────────────────────────────────────────────────
OUTPUT_DIR = '/home/baruva/network_cache/olcr_data'  # output directory for interferograms and metadata

# Network sweep parameters
NUM_EXTERNAL_NODES = 1
NI_VALUES          = [4, 6, 12]       # internal-node counts to sweep

# Seed strategy: seed = config_id (varies per configuration)
CONFIG_ID_START    = 42
CONFIG_ID_END      = 43               # exclusive

# OLCR parameters
CENTRAL_WAVELENGTH    = 1000e-9
BANDWIDTH             = 400e-9
INTEGRATION_METHOD    = 'czt'
WAVELENGTH_SAMPLING   = int(1e6)
MAX_OPL               = 2000e-6
MAX_LENGTH_SAMPLING   = int(1e5)
USE_MP                = True

# Fault placement
FAULT_START    = 1e-6                 # first fault position
FAULT_SPACING  = 1e-6                 # spacing between faults

# Network spec (fixed across sweep)
NETWORK_TYPE    = 'delaunay'
NETWORK_SHAPE   = 'slab'
NETWORK_SIZE    = (200e-6, 200e-6)
EXTERNAL_OFFSET = 10e-6
FULLY_CONNECTED = True
NODE_S_MAT_TYPE = 'neumann'
# ──────────────────────────────────────────────────────────────────────────────


def generate_interferograms_for_config(config_id, spec, output_dir):
    """
    Generate reference and perturbed interferograms for a single network
    configuration, with caching: skips entirely if metadata already exists.

    Returns (fingerprint, elapsed_time, total_perturbations).
    """
    print(f"\n=== Configuration {config_id}  (ni={spec.num_internal_nodes}) ===")
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # ── Caching check ─────────────────────────────────────────────────────────
    fingerprint   = _spec_fingerprint(spec)
    metadata_file = os.path.join(output_dir, f'{fingerprint}_metadata.npy')

    if os.path.exists(metadata_file):
        print(f"  Skipping: cached metadata found → {metadata_file}")
        try:
            md = np.load(metadata_file, allow_pickle=True).item()
            return fingerprint, md.get('generation_time', 0), md.get('total_perturbations', 0)
        except Exception:
            return fingerprint, 0, 0
    # ─────────────────────────────────────────────────────────────────────────

    network = generate_network(spec)

    # ── Reference interferograms ──────────────────────────────────────────────
    for source_idx in [0, 1]:
        print(f"  Computing reference interferogram for source {source_idx + 1}...")
        olcr_ref = OLCR(
            network=network,
            input_node=source_idx,
            measurement_node=source_idx,
            central_wavelength=CENTRAL_WAVELENGTH,
            bandwidth=BANDWIDTH,
            integeration_method=INTEGRATION_METHOD,
            num_wavelength_sample=WAVELENGTH_SAMPLING,
            optical_path_length=[0, MAX_OPL],
            num_optical_path_length_sample=MAX_LENGTH_SAMPLING,
            use_multi_proc=USE_MP,
        )

        I_ref = olcr_ref.get_interferogram()
        np.save(os.path.join(output_dir, f'{fingerprint}_ref_s{source_idx + 1}.npy'), I_ref)

        if source_idx == 0:
            np.save(os.path.join(output_dir, f'{fingerprint}_opls.npy'), olcr_ref.opls)

    # ── Perturbed interferograms ──────────────────────────────────────────────
    links              = [link.index for link in network.internal_links]
    total_perturbations = 0

    for link_index in links:
        print(f"    Processing link {link_index}...")
        link_length    = network.get_link(link_index).length
        fault_positions = np.arange(FAULT_START, link_length, FAULT_SPACING)
        ratios          = np.round(fault_positions / link_length, 2)
        ratios          = np.unique(ratios)
        ratios          = ratios[(ratios > 0) & (ratios < 1)]

        for ratio in ratios:
            perturbator      = NetworkPerturbator(network)
            perturbator.add_perturbation_node(link_index=link_index, fractional_position=ratio)
            perturbed_network = perturbator.perturbed_network

            for source_idx in [0, 1]:
                olcr_pert = OLCR(
                    network=perturbed_network,
                    input_node=source_idx,
                    measurement_node=source_idx,
                    central_wavelength=CENTRAL_WAVELENGTH,
                    bandwidth=BANDWIDTH,
                    integeration_method=INTEGRATION_METHOD,
                    num_wavelength_sample=WAVELENGTH_SAMPLING,
                    optical_path_length=[0, MAX_OPL],
                    num_optical_path_length_sample=MAX_LENGTH_SAMPLING,
                    use_multi_proc=USE_MP,
                )

                I_pert = olcr_pert.get_interferogram()
                fname  = f'{fingerprint}_pert_link{link_index}_r{ratio:.2f}_s{source_idx + 1}.npy'
                np.save(os.path.join(output_dir, fname), I_pert)

            total_perturbations += 1

    # ── Metadata ──────────────────────────────────────────────────────────────
    elapsed_time = time.time() - start_time
    metadata = {
        'config_id':            config_id,
        'fingerprint':          fingerprint,
        'wavelength_sampling':  WAVELENGTH_SAMPLING,
        'max_opl':              MAX_OPL,
        'max_length_sampling':  MAX_LENGTH_SAMPLING,
        'central_wavelength':   CENTRAL_WAVELENGTH,
        'bandwidth':            BANDWIDTH,
        'integration_method':   INTEGRATION_METHOD,
        'num_nodes':            len(network.nodes),
        'num_links':            len(network.links),
        'num_external_nodes':   len(network.external_nodes),
        'num_internal_nodes':   len(network.internal_nodes),
        'num_internal_links':   len(network.internal_links),
        'total_perturbations':  total_perturbations,
        'generation_time':      elapsed_time,
        'random_seed':          spec.random_seed,
        'network_spec': {
            'num_external_nodes': spec.num_external_nodes,
            'num_internal_nodes': spec.num_internal_nodes,
            'network_type':       spec.network_type,
            'network_shape':      spec.network_shape,
            'network_size':       spec.network_size,
            'external_offset':    spec.external_offset,
            'fully_connected':    spec.fully_connected,
            'node_S_mat_type':    spec.node_S_mat_type,
        },
    }
    np.save(metadata_file, metadata)

    print(f"  Done in {elapsed_time:.2f}s — {total_perturbations} perturbations across {len(links)} links")
    return fingerprint, elapsed_time, total_perturbations


def main():
    print("Starting interferogram generation...")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"NI values        : {NI_VALUES}")
    print(f"Config ID range  : [{CONFIG_ID_START}, {CONFIG_ID_END})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()

    grand_total_perturbations = 0
    grand_successful          = 0
    grand_failed              = []

    for ni in NI_VALUES:
        print(f"\n{'='*60}")
        print(f"  Sweeping ni = {ni}")
        print(f"{'='*60}")

        sweep_perturbations = 0
        sweep_successful    = 0
        sweep_failed        = []

        for config_id in range(CONFIG_ID_START, CONFIG_ID_END):
            try:
                spec = NetworkSpec(
                    num_external_nodes=NUM_EXTERNAL_NODES,
                    num_internal_nodes=ni,
                    network_type=NETWORK_TYPE,
                    network_shape=NETWORK_SHAPE,
                    network_size=NETWORK_SIZE,
                    external_offset=EXTERNAL_OFFSET,
                    random_seed=config_id,      # seed varies with config_id
                    fully_connected=FULLY_CONNECTED,
                    node_S_mat_type=NODE_S_MAT_TYPE,
                )

                fingerprint, elapsed, n_pert = generate_interferograms_for_config(
                    config_id, spec, OUTPUT_DIR
                )

                sweep_successful    += 1
                sweep_perturbations += n_pert
                print(f"  ✓ config {config_id} | ni={ni} | {n_pert} perturbations | {elapsed:.1f}s")

            except Exception as e:
                print(f"  ✗ config {config_id} | ni={ni} | ERROR: {e}")
                sweep_failed.append((ni, config_id))

        grand_total_perturbations += sweep_perturbations
        grand_successful          += sweep_successful
        grand_failed              += sweep_failed

        print(f"\n  ni={ni} sweep complete: {sweep_successful} configs, "
              f"{sweep_perturbations} perturbations, {len(sweep_failed)} failures")

    total_time = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"  ALL SWEEPS COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time          : {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Successful configs  : {grand_successful}")
    print(f"  Total perturbations : {grand_total_perturbations}")
    if grand_failed:
        print(f"  Failed (ni, config) : {grand_failed}")

    summary = {
        'total_time':               total_time,
        'successful_configs':       grand_successful,
        'total_perturbations':      grand_total_perturbations,
        'failed_configs':           grand_failed,
        'ni_values':                NI_VALUES,
        'config_id_range':          (CONFIG_ID_START, CONFIG_ID_END),
        'parameters': {
            'ne':               NUM_EXTERNAL_NODES,
            'ni_values':        NI_VALUES,
            'network_type':     NETWORK_TYPE,
            'network_shape':    NETWORK_SHAPE,
            'network_size':     NETWORK_SIZE,
            'external_offset':  EXTERNAL_OFFSET,
            'fully_connected':  FULLY_CONNECTED,
            'node_S_mat_type':  NODE_S_MAT_TYPE,
            'max_opl':          MAX_OPL,
            'fault_start':      FAULT_START,
            'fault_spacing':    FAULT_SPACING,
        },
    }
    np.save(os.path.join(OUTPUT_DIR, 'generation_summary.npy'), summary)
    print(f"  Summary saved to    : {OUTPUT_DIR}/generation_summary.npy")


if __name__ == "__main__":
    main()