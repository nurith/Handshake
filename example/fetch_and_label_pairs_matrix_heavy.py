#!/usr/bin/env python3
"""
fetch_and_label_pairs.py

Reads a text file where each line is a protein pair ID like:
    3TXS_C_3TXS_B

For each pair:
  1. Fetches the PDB structure using BioPython
  2. Extracts sequences for both chains
  3. Labels each residue 1 if any C-alpha in the OTHER chain is within
     6 Angstroms, 0 otherwise
  4. Writes a CSV with columns:
     pair_id, chain_A, chain_B, seq_A, seq_B, label_A, label_B

Usage:
    python fetch_and_label_pairs.py pairs.txt --output labeled_pairs.csv
    python fetch_and_label_pairs.py pairs.txt --output labeled_pairs.csv --threshold 8.0
    python fetch_and_label_pairs.py pairs.txt --output labeled_pairs.csv --pdb-dir ./pdb_cache

Requirements:
    pip install biopython pandas tqdm
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# BioPython imports
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

warnings.filterwarnings("ignore")  # suppress BioPython PDB warnings

# ── Constants ──────────────────────────────────────────────────────────────────

THRESHOLD_ANGSTROM = 6.0      # C-alpha distance cutoff for binding site label
PDB_DOWNLOAD_DIR   = "./pdb_cache"   # local cache for downloaded PDB files
RETRY_ATTEMPTS     = 3        # number of download retries per structure
RETRY_DELAY        = 5        # seconds between retries


# ── PDB fetching ───────────────────────────────────────────────────────────────

def fetch_structure(pdb_id, pdb_dir, use_mmcif=False):
    """
    Download and parse a PDB structure (asymmetric unit only).
    Downloads directly from RCSB by URL — bypasses BioPython PDBList
    which can fetch biological assembly files with transformed coordinates
    that cause all residues to appear within the distance threshold.

    Returns a BioPython Structure object or None on failure.
    """
    import urllib.request

    pdb_id  = pdb_id.lower()
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    cif_path = pdb_dir / f"{pdb_id}.cif"

    # Check cache
    if pdb_path.exists():
        try:
            return PDBParser(QUIET=True).get_structure(pdb_id, str(pdb_path))
        except Exception:
            pdb_path.unlink()   # corrupt — delete and re-download

    if cif_path.exists():
        try:
            return MMCIFParser(QUIET=True).get_structure(pdb_id, str(cif_path))
        except Exception:
            cif_path.unlink()

    # Download asymmetric unit directly from RCSB by URL.
    # These always return deposited coordinates with original chain IDs,
    # never the biological assembly.
    urls = [
        (f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", pdb_path, "pdb"),
        (f"https://files.rcsb.org/download/{pdb_id.upper()}.cif", cif_path, "cif"),
    ]

    for url, dest, fmt in urls:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                urllib.request.urlretrieve(url, str(dest))
                if fmt == "pdb":
                    return PDBParser(QUIET=True).get_structure(pdb_id, str(dest))
                else:
                    return MMCIFParser(QUIET=True).get_structure(pdb_id, str(dest))
            except Exception as e:
                if dest.exists():
                    dest.unlink()
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    break   # try next format

    print(f"  [WARN] Failed to fetch {pdb_id} from RCSB")
    return None


# ── Sequence and coordinate extraction ────────────────────────────────────────

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Non-standard residues mapped to X
}


def get_chain_residues(structure, chain_id):
    """
    Extract standard amino acid residues from a chain.
    Returns list of (residue_object, one_letter_code) tuples,
    skipping heteroatoms (HETATM) and water.
    """
    residues = []
    for model in structure:
        if chain_id not in [c.id for c in model]:
            continue
        chain    = model[chain_id]
        seen_res = set()   # track (res_seq, ins_code) to skip duplicates

        for residue in chain:
            # Skip heteroatoms (water, ligands) — only standard AA
            if residue.id[0] != ' ':
                continue
            if not is_aa(residue, standard=False):
                continue

            # Skip alternate conformations — keep only the primary one.
            res_key = (residue.id[1], residue.id[2])  # (seq_num, ins_code)
            if res_key in seen_res:
                continue
            seen_res.add(res_key)

            # Skip residues with no C-alpha — NaN coords corrupt the distance
            # matrix (nansum returns 0, making distance appear as 0 Å)
            if 'CA' not in residue:
                continue

            resname = residue.get_resname().strip()
            aa = THREE_TO_ONE.get(resname, 'X')
            residues.append((residue, aa))
        break  # use first model only
    return residues


def get_calpha_coords(residues):
    """
    Extract C-alpha coordinates as (N, 3) numpy array.
    All residues are guaranteed to have CA (filtered in get_chain_residues).
    """
    return np.array([
        residue['CA'].get_vector().get_array()
        for residue, _ in residues
    ])


def get_heavy_atom_coords(residues):
    """
    Extract heavy-atom (non-hydrogen) coordinates per residue.

    Returns a list of length N where each entry is an (M_i, 3) numpy array
    of heavy-atom coordinates for residue i. Residues vary in atom count
    so we cannot use a single 2D array.
    """
    out = []
    for residue, _ in residues:
        atoms = []
        for atom in residue.get_atoms():
            element = atom.element.strip().upper() if atom.element else ''
            # Skip hydrogens (element 'H' or 'D' for deuterium)
            if element in ('H', 'D'):
                continue
            # Fallback for files without element tags: skip names starting with H
            if not element and atom.get_name().strip().startswith('H'):
                continue
            atoms.append(atom.get_vector().get_array())
        if atoms:
            out.append(np.array(atoms))
        else:
            # Fallback to CA if no heavy atoms found (should not happen)
            out.append(np.array([residue['CA'].get_vector().get_array()]))
    return out


# ── Labelling ─────────────────────────────────────────────────────────────────

def label_binding_sites(coords_a, coords_b, threshold=THRESHOLD_ANGSTROM):
    """
    Label each residue in chain A as 1 if ANY C-alpha in chain B is within
    `threshold` Angstroms, 0 otherwise. Same for chain B vs chain A.
    Also returns the sparse contact list as a list of (i, j) pairs.

    Returns:
        labels_a : list of int (0 or 1), length = len(coords_a)
        labels_b : list of int (0 or 1), length = len(coords_b)
        contacts : list of (int, int) — (i, j) index pairs in contact
    """
    n_a = len(coords_a)
    n_b = len(coords_b)

    labels_a = [0] * n_a
    labels_b = [0] * n_b
    contacts  = []
  
    if n_a == 0 or n_b == 0:
        return labels_a, labels_b, contacts

    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))   # (n_a, n_b)

    within_thresh = dist < threshold
    labels_a = within_thresh.any(axis=1).astype(int).tolist()
    labels_b = within_thresh.any(axis=0).astype(int).tolist()
    ij = np.argwhere(within_thresh)
    contacts = [(int(i), int(j)) for i, j in ij]
    return labels_a, labels_b, contacts


def label_binding_sites_heavy(heavy_a, heavy_b, threshold=THRESHOLD_ANGSTROM,
                              ca_a=None, ca_b=None, ca_cutoff_pad=12.0):
    """
    Heavy-atom version: residue i in chain A is labeled 1 if any
    non-hydrogen atom in residue i is within `threshold` Angstroms of any
    non-hydrogen atom in any residue of chain B.

    Optimization: if Cα coordinates are provided, residue pairs whose
    Cα distance exceeds `threshold + ca_cutoff_pad` Å are skipped, since
    no heavy atom can be closer than (Cα distance − pad). Default pad of
    12 Å is conservative — typical residue Cα-to-side-chain extents are
    7–8 Å, so 12 Å covers Arg/Lys/Trp/Tyr without missing any interactions.

    Args:
        heavy_a : list of (M_i, 3) arrays of heavy-atom coords per residue (chain A)
        heavy_b : list of (M_j, 3) arrays of heavy-atom coords per residue (chain B)
        threshold : heavy-atom distance threshold (default 6.0 Å for PPInt)
        ca_a, ca_b : (N, 3) Cα coords for early termination. If None, no skip.
        ca_cutoff_pad : how much beyond `threshold` the Cα distance must exceed
                        before we skip the residue pair. Larger = safer/slower.

    Returns:
        labels_a, labels_b, contacts (same format as label_binding_sites)
    """
    n_a = len(heavy_a)
    n_b = len(heavy_b)

    labels_a = [0] * n_a
    labels_b = [0] * n_b
    contacts = []

    if n_a == 0 or n_b == 0:
        return labels_a, labels_b, contacts

    # Sanity check: per-residue heavy-atom lists must align with expected lengths
    assert len(heavy_a) == n_a, f"heavy_a length mismatch: {len(heavy_a)} vs {n_a}"
    assert len(heavy_b) == n_b, f"heavy_b length mismatch: {len(heavy_b)} vs {n_b}"

    # Optional Cα-based skip mask
    if ca_a is not None and ca_b is not None:
        ca_diff = ca_a[:, np.newaxis, :] - ca_b[np.newaxis, :, :]
        ca_dist = np.sqrt((ca_diff ** 2).sum(axis=2))
        skip_pair = ca_dist > (threshold + ca_cutoff_pad)
    else:
        skip_pair = np.zeros((n_a, n_b), dtype=bool)

    # Pairwise minimum heavy-atom distance per residue pair (i, j).
    # Initialized to inf so skipped pairs are guaranteed > threshold.
    min_dist = np.full((n_a, n_b), np.inf, dtype=np.float32)

    for i in range(n_a):
        atoms_i = heavy_a[i]                      # (M_i, 3)
        for j in range(n_b):
            if skip_pair[i, j]:
                continue                           # Cα too far — no heavy atom can be close
            atoms_j = heavy_b[j]                  # (M_j, 3)
            diff = atoms_i[:, np.newaxis, :] - atoms_j[np.newaxis, :, :]
            d = np.sqrt((diff ** 2).sum(axis=2)).min()
            min_dist[i, j] = d

    within_thresh = min_dist < threshold
    labels_a = within_thresh.any(axis=1).astype(int).tolist()
    labels_b = within_thresh.any(axis=0).astype(int).tolist()
    ij = np.argwhere(within_thresh)
    contacts = [(int(i), int(j)) for i, j in ij]

    # Sanity check: contact indices are within bounds (should always hold by construction)
    if contacts:
        max_i = max(i for i, _ in contacts)
        max_j = max(j for _, j in contacts)
        assert max_i < n_a and max_j < n_b, \
            f"Contact index out of bounds: ({max_i}, {max_j}) for ({n_a}, {n_b})"

    return labels_a, labels_b, contacts


# ── Parse pair ID ──────────────────────────────────────────────────────────────

def parse_pair_id(line):
    """
    Parse a line like '3TXS_C_3TXS_B' into (pdb_id, chain_a, chain_b).
    Both chains must come from the same PDB entry (pdb_id_A == pdb_id_B).
    Also handles lines with comments: '3TXS_C_3TXS_B  # some comment'
    """
    line = line.strip().split('#')[0].strip()   # strip comments
    if not line:
        return None

    parts = line.split('_')
    # Format: XXXX_C_XXXX_B  → 4 parts
    if len(parts) != 4:
        print(f"  [WARN] Unexpected format: '{line}' — expected XXXX_C_XXXX_B, skipping")
        return None

    pdb_a, chain_a, pdb_b, chain_b = parts

    if pdb_a.upper() != pdb_b.upper():
        print(f"  [WARN] Cross-PDB pair '{line}' — PDB IDs differ ({pdb_a} vs {pdb_b}), skipping")
        return None

    return pdb_a.upper(), chain_a.upper(), chain_b.upper()


# ── Resolution check ──────────────────────────────────────────────────────────

def check_resolution(structure, max_resolution, exclude_nmr):
    """
    Check whether structure passes resolution filtering criteria.

    Returns:
        (ok, reason): ok is True if structure should be kept, False otherwise.
                      reason is a short string for the rejection log.

    Behavior:
      - X-ray and cryo-EM structures: rejected if resolution > max_resolution
        or if resolution is missing.
      - NMR structures: rejected if exclude_nmr=True (no resolution metric).
      - Other methods: kept (rare; e.g. "ELECTRON CRYSTALLOGRAPHY").

    Header fields read:
      structure.header['resolution']   — float or None
      structure.header['structure_method'] — string (e.g., 'x-ray diffraction',
                                              'electron microscopy',
                                              'solution nmr')
    """
    header = structure.header or {}
    method = (header.get('structure_method') or '').lower().strip()
    res    = header.get('resolution')

    is_nmr = 'nmr' in method
    if is_nmr:
        if exclude_nmr:
            return False, f"NMR structure (method='{method}')"
        return True, "NMR structure (kept)"

    # X-ray, cryo-EM, etc. — require resolution and apply cutoff
    if res is None:
        return False, f"no resolution reported (method='{method}')"

    try:
        res_val = float(res)
    except (TypeError, ValueError):
        return False, f"unparseable resolution '{res}' (method='{method}')"

    if res_val > max_resolution:
        return False, f"resolution {res_val:.2f}A > {max_resolution:.2f}A cutoff (method='{method}')"

    return True, f"resolution {res_val:.2f}A (method='{method}')"


# ── Main processing ────────────────────────────────────────────────────────────

def process_pair(pair_id_str, pdb_dir, threshold,
                 min_length=50, max_length=300, distance_mode='calpha',
                 max_resolution=None, exclude_nmr=False):
    """
    Process one pair string. Returns a dict row or None on failure.
    Pairs where either chain has fewer than min_length or more than
    max_length residues (after filtering residues with no CA) are discarded.

    distance_mode: 'calpha' uses C-alpha to C-alpha distance (fast).
                   'heavy' uses minimum non-hydrogen atom distance (slower
                   but matches the PPInt convention).
    """
    parsed = parse_pair_id(pair_id_str)
    if parsed is None:
        return None

    pdb_id, chain_a, chain_b = parsed

    structure = fetch_structure(pdb_id, pdb_dir)
    if structure is None:
        return None

    # Resolution / experimental method filter
    if max_resolution is not None:
        ok, reason = check_resolution(structure, max_resolution, exclude_nmr)
        if not ok:
            print(f"  [SKIP] {pdb_id}: {reason}")
            return None

    # Extract residues for both chains
    residues_a = get_chain_residues(structure, chain_a)
    residues_b = get_chain_residues(structure, chain_b)

    if not residues_a:
        print(f"  [WARN] {pdb_id}: chain {chain_a} not found or empty")
        print(f"  Empty: {pair_id_str}")
        return None
    if not residues_b:
        print(f"  [WARN] {pdb_id}: chain {chain_b} not found or empty")
        print(f"  Empty: {pair_id_str}")
        return None

    # Length filters
    if len(residues_a) < min_length:
        print(f"  [SKIP] {pdb_id}: chain {chain_a} too short ({len(residues_a)} < {min_length})")
        if(len(residues_a) < 30):
           print(f" Short: {pair_id_str}")
        return None
    if len(residues_b) < min_length:
        print(f"  [SKIP] {pdb_id}: chain {chain_b} too short ({len(residues_b)} < {min_length})")
        if(len(residues_b) < 30):
           print(f" Short: {pair_id_str}")
        return None
    if len(residues_a) > max_length:
        if(len(residues_a) > 500):
           print(f" Long: {pair_id_str}")
        print(f"  [SKIP] {pdb_id}: chain {chain_a} too long ({len(residues_a)} > {max_length})")
        return None
    if len(residues_b) > max_length:
        if(len(residues_b) > 500):
           print(f" Long: {pair_id_str}")
        print(f"  [SKIP] {pdb_id}: chain {chain_b} too long ({len(residues_b)} > {max_length})")
        return None

    seq_a = ''.join(aa for _, aa in residues_a)
    seq_b = ''.join(aa for _, aa in residues_b)

    if distance_mode == 'heavy':
        heavy_a = get_heavy_atom_coords(residues_a)
        heavy_b = get_heavy_atom_coords(residues_b)
        # Cα coords are cheap to compute and enable early termination
        ca_a = get_calpha_coords(residues_a)
        ca_b = get_calpha_coords(residues_b)
        labels_a, labels_b, contacts = label_binding_sites_heavy(
            heavy_a, heavy_b, threshold, ca_a=ca_a, ca_b=ca_b)
    else:
        coords_a = get_calpha_coords(residues_a)
        coords_b = get_calpha_coords(residues_b)
        labels_a, labels_b, contacts = label_binding_sites(coords_a, coords_b, threshold)

    # Sanity check: label arrays must match sequence lengths
    assert len(labels_a) == len(seq_a), \
        f"label_A/seq_A length mismatch: {len(labels_a)} vs {len(seq_a)} for {pair_id_str}"
    assert len(labels_b) == len(seq_b), \
        f"label_B/seq_B length mismatch: {len(labels_b)} vs {len(seq_b)} for {pair_id_str}"

    binding_a  = sum(labels_a)
    binding_b  = sum(labels_b)
    pct_a      = round(binding_a / len(seq_a) * 100, 1) if seq_a else 0
    pct_b      = round(binding_b / len(seq_b) * 100, 1) if seq_b else 0

    # Warn if suspiciously high — may indicate a bad cached PDB file
    if pct_a == 100.0 or pct_b == 100.0:
        print(f"  [WARN] {pdb_id} {chain_a}/{chain_b}: "
              f"100% binding ({pct_a}%/{pct_b}%) — "
              f"possibly a stale biological assembly in cache. "
              f"Delete {pdb_id}.pdb/.cif from pdb_dir and retry.")

    # Encode contacts as compact string: "i1,j1;i2,j2;..."
    # Each (i,j) means residue i in chain A contacts residue j in chain B
    contacts_str = ";".join(f"{i},{j}" for i, j in contacts)

    return {
        'pair_id':       f"{pdb_id}_{chain_a}_{pdb_id}_{chain_b}",
        'pdb_id':        pdb_id,
        'chain_A':       f"{pdb_id}_{chain_a}",
        'chain_B':       f"{pdb_id}_{chain_b}",
        'seq_A':         seq_a,
        'seq_B':         seq_b,
        'label_A':       ''.join(str(l) for l in labels_a),
        'label_B':       ''.join(str(l) for l in labels_b),
        'contacts':      contacts_str,
        'len_A':         len(seq_a),
        'len_B':         len(seq_b),
        'binding_A':     binding_a,
        'binding_B':     binding_b,
        'n_contacts':    len(contacts),
        'pct_binding_A': pct_a,
        'pct_binding_B': pct_b,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch PDB structures and label binding sites by C-alpha distance"
    )
    parser.add_argument("input",
                        help="Text file with one pair ID per line (e.g. 3TXS_C_3TXS_B)")
    parser.add_argument("--output", "-o", default="labeled_pairs.csv",
                        help="Output CSV file (default: labeled_pairs.csv)")
    parser.add_argument("--threshold", "-t", type=float, default=THRESHOLD_ANGSTROM,
                        help=f"Distance threshold in Angstroms (default: {THRESHOLD_ANGSTROM})")
    parser.add_argument("--distance-mode", choices=['calpha', 'heavy'], default='calpha',
                        help="Distance calculation mode: 'calpha' uses C-alpha to "
                             "C-alpha distance (fast). 'heavy' uses minimum non-hydrogen "
                             "atom distance per residue pair (slower, matches PPInt convention).")
    parser.add_argument("--max-resolution", type=float, default=None,
                        help="Maximum resolution in Angstroms for X-ray/cryo-EM structures "
                             "(default: no filtering). Suggested: 4.0 for typical use, "
                             "3.5 for stricter filtering. Structures missing a reported "
                             "resolution are excluded when this flag is set.")
    parser.add_argument("--exclude-nmr", action='store_true',
                        help="Exclude NMR-determined structures, which lack a comparable "
                             "resolution metric and may have ensemble-averaging artifacts.")
    parser.add_argument("--min-length", type=int, default=50,
                        help="Minimum chain length in residues (default: 50)")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum chain length in residues (default: 300)")
    parser.add_argument("--max-binding-pct", type=float, default=80.0,
                        help="Exclude pairs where either chain exceeds this %% binding (default: 80)")
    parser.add_argument("--min-binding-pct", type=float, default=1.0,
                        help="Exclude pairs where either chain has less than this %% binding (default: 1.0)")
    parser.add_argument("--pdb-dir", default=PDB_DOWNLOAD_DIR,
                        help=f"Directory for cached PDB files (default: {PDB_DOWNLOAD_DIR})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pairs already present in output file (useful for resuming)")
    parser.add_argument("--missing", "-m", default="missing.txt",
                        help="Missing or otherwise defunct for future removal")
    args = parser.parse_args()

    # Read pair IDs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    print(f"Found {len(lines)} pairs in {args.input}")
    print(f"Distance mode:     {args.distance_mode} ({'C-α to C-α' if args.distance_mode == 'calpha' else 'min heavy-atom'})")
    print(f"Distance threshold:{args.threshold} Å")
    if args.max_resolution is not None:
        print(f"Max resolution:    {args.max_resolution} Å (X-ray/cryo-EM only)")
    else:
        print(f"Max resolution:    no filter (all resolutions included)")
    print(f"NMR structures:    {'excluded' if args.exclude_nmr else 'included'}")
    print(f"Chain length:      {args.min_length}–{args.max_length} residues")
    print(f"Binding%%:          {args.min_binding_pct}%–{args.max_binding_pct}%% per chain")
    print(f"PDB cache dir:     {args.pdb_dir}")
    print(f"Output:            {args.output}")
    print()

    # Load existing results if resuming
    existing_ids = set()
    if args.skip_existing and Path(args.output).exists():
        existing_df = pd.read_csv(args.output)
        existing_ids = set(existing_df['pair_id'].tolist())
        print(f"Resuming: {len(existing_ids)} pairs already processed")

    # Process pairs
    results   = []
    skipped   = 0
    failed    = 0
    write_header = not (args.skip_existing and Path(args.output).exists())
    miss = open(args.missing, 'w')
    
    with open(args.output, 'a' if args.skip_existing else 'w', newline='') as csvfile:
        fieldnames = ['pair_id', 'pdb_id', 'chain_A', 'chain_B',
                      'seq_A', 'seq_B', 'label_A', 'label_B', 'contacts',
                      'len_A', 'len_B', 'binding_A', 'binding_B',
                      'n_contacts', 'pct_binding_A', 'pct_binding_B']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for line in tqdm(lines, desc="Processing pairs"):
            parsed = parse_pair_id(line)
            if parsed is None:
                skipped += 1
                continue

            pdb_id, chain_a, chain_b = parsed
            pair_key = f"{pdb_id}_{chain_a}_{pdb_id}_{chain_b}"
            if pair_key in existing_ids:
                skipped += 1
                continue

            row = process_pair(line, args.pdb_dir, args.threshold,
                               min_length=args.min_length,
                               max_length=args.max_length,
                               distance_mode=args.distance_mode,
                               max_resolution=args.max_resolution,
                               exclude_nmr=args.exclude_nmr)
            if row is None:
                failed += 1
                continue

            # Filter out suspicious all-interacting pairs
            if (row['pct_binding_A'] > args.max_binding_pct or
                    row['pct_binding_B'] > args.max_binding_pct):
                print(f"  [FILTER] {pair_key}: "
                      f"{row['pct_binding_A']}%/{row['pct_binding_B']}% binding "
                      f"> {args.max_binding_pct}% — excluded (stale cache?)")
                failed += 1
                continue

            # Filter out pairs with no binding sites on either chain
            if (row['pct_binding_A'] < args.min_binding_pct or
                    row['pct_binding_B'] < args.min_binding_pct):
                print(f"  [FILTER] {pair_key}: "
                      f"{row['pct_binding_A']}%/{row['pct_binding_B']}% binding "
                      f"< {args.min_binding_pct}% — excluded (no interface detected)")
                miss.write(pair_key)
                miss.write("\n")
                failed += 1
                continue

            writer.writerow(row)
            csvfile.flush()
            results.append(row)

    # Summary
    total_processed = len(results)
    print(f"\n── Summary ───────────────────────────────────────")
    print(f"Processed:  {total_processed}")
    print(f"Skipped:    {skipped}  (bad format or already done)")
    print(f"Failed/filtered: {failed}")

    if results:
        df = pd.DataFrame(results)
        print(f"Avg binding% chain A: {df['pct_binding_A'].mean():.1f}%")
        print(f"Avg binding% chain B: {df['pct_binding_B'].mean():.1f}%")
        n_high = ((df['pct_binding_A'] > 50) | (df['pct_binding_B'] > 50)).sum()
        if n_high:
            print(f"Pairs with >50%% binding: {n_high} (may warrant inspection)")
        print(f"\nOutput saved to: {args.output}")
        print(f"\nTip: if you see 100%% binding cases, delete the corresponding")
        print(f"     .pdb/.cif files from {args.pdb_dir} and rerun with --skip-existing")


if __name__ == "__main__":
    main()
