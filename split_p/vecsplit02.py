# Refactored vecsplit01_notebook.ipynb

# --- Cell 1: Imports & Configuration ----------------------------------------
import argparse
from pathlib import Path
import logging
import os
import importlib

# Dependency version checks
REQUIRED_LIBS = {
    'pandas': '1.3.0',
    'numpy': '1.21.0',
    'torch': '1.11.0',
    'transformers': '4.18.0',
    'openpyxl': '3.0.0'
}

_missing_versions = []
for lib, min_ver in REQUIRED_LIBS.items():
    try:
        mod = importlib.import_module(lib)
        installed_ver = getattr(mod, '__version__', '0')
        from packaging import version
        if version.parse(installed_ver) < version.parse(min_ver):
            _missing_versions.append(f"{lib}>={min_ver} (installed: {installed_ver})")
    except ImportError:
        _missing_versions.append(f"{lib}>={min_ver} (not installed)")

if _missing_versions:
    missing_str = "; ".join(_missing_versions)
    raise ImportError(f"Missing or outdated dependencies: {missing_str}. "
                      "Please install via `pip install -r requirements.txt`. ")

# Standard imports
import numpy as np
import torch

# Local modules
from vecsplit.io_manager import load_excel, write_excel
from vecsplit.tokenizer import split_src_meaning_units, split_tgt_meaning_units
from vecsplit.embedder import compute_embeddings
from vecsplit.aligner import align_pairs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define max_workers only once globally
DEFAULT_MAX_WORKERS = os.cpu_count() or 4

# --- Cell 2: CLI Argument Parsing -------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            input: Path to input .xlsx
            output: Path to output .xlsx
            batch_size: Embedding batch size
            max_workers: Number of threads for tokenization
            device: torch device ('cpu' or 'cuda')
    """
    parser = argparse.ArgumentParser(
        description="Parallel alignment pipeline for classical Chinese and Korean translations"
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to input Excel file (.xlsx)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Path to output Excel file (.xlsx)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Number of worker threads for tokenization (default: CPU count)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Computation device for embeddings"
    )
    return parser.parse_args()

# --- Cell 3: Main Pipeline ---------------------------------------------------
def main() -> None:
    """
    Execute the full pipeline: load data, split into meaning units,
    compute embeddings, align pairs, and write output.
    """
    import pandas as pd  # local scope import to avoid global namespace pollution

    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    logger.info(f"Loading data from {input_path}")
    df = load_excel(input_path)

    # Validate input
    required_cols = {"원문", "번역문"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise KeyError(f"Input file missing required columns: {missing_cols}")

    texts_src = df["원문"].astype(str).tolist()
    texts_tgt = df["번역문"].astype(str).tolist()

    # Split into meaning units
    logger.info("Splitting source text into meaning units...")
    src_units = split_src_meaning_units(
        texts_src, max_workers=args.max_workers
    )
    logger.info("Splitting target text into meaning units...")
    tgt_units = split_tgt_meaning_units(
        texts_tgt, max_workers=args.max_workers
    )

    # Compute embeddings
    logger.info(f"Computing embeddings (batch_size={args.batch_size}, device={args.device})")
    src_emb = compute_embeddings(
        src_units, batch_size=args.batch_size, device=args.device
    )
    tgt_emb = compute_embeddings(
        tgt_units, batch_size=args.batch_size, device=args.device
    )

    # Align sequences
    logger.info("Running DP-based alignment algorithm...")
    aligned = align_pairs(src_units, tgt_units, src_emb, tgt_emb)

    # Write results
    logger.info(f"Writing results to {output_path}")
    write_excel(output_path, aligned)
    logger.info("Pipeline completed successfully.")

def align_by_similarity(src_units, tgt_units, model, sim_threshold=0.35):
    if not src_units or not tgt_units:
        return []
    batch_size = min(32, max(8, len(src_units) + len(tgt_units)//4))
    # BGE-M3 임베딩
    src_out = model.encode(src_units, batch_size=batch_size, return_dense=True, normalize_embeddings=True)
    tgt_out = model.encode(tgt_units, batch_size=batch_size, return_dense=True, normalize_embeddings=True)
    # 버그 수정: 'dense_vecs'가 numpy array가 아니라 torch tensor일 수 있으므로 numpy 변환 추가
    src_vecs = src_out['dense_vecs']
    tgt_vecs = tgt_out['dense_vecs']
    if hasattr(src_vecs, "cpu"):
        src_vecs = src_vecs.cpu().numpy()
    if hasattr(tgt_vecs, "cpu"):
        tgt_vecs = tgt_vecs.cpu().numpy()
    sim = cosine_similarity(src_vecs, tgt_vecs)
    n, m = sim.shape

    dp = np.full((n + 1, m + 1), -np.inf)
    bt = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, 0] = 0

    def score(i, j):
        similarity = sim[i, j]
        len_src = len(src_units[i])
        len_tgt = len(tgt_units[j])
        len_ratio = min(len_src, len_tgt) / max(len_src, len_tgt) if max(len_src, len_tgt) > 0 else 0
        hanja_ratio_src = len(re.findall(r'\p{Han}', src_units[i])) / len(src_units[i]) if len(src_units[i]) > 0 else 0
        hangul_ratio_tgt = len(re.findall(r'[\uAC00-\uD7A3]', tgt_units[j])) / len(tgt_units[j]) if len(tgt_units[j]) > 0 else 0
        char_bonus = 0.1 * (hanja_ratio_src + hangul_ratio_tgt) / 2
        final_score = similarity * (0.8 + 0.2 * len_ratio) + char_bonus
        return final_score if final_score >= sim_threshold else -1.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m:
                s = score(i, j)
                if dp[i, j] + s > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = dp[i, j] + s
                    bt[i + 1, j + 1] = 1
            if i < n and dp[i, j] > dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j]
                bt[i + 1, j] = 2
            if j < m and dp[i, j] > dp[i, j + 1]:
                dp[i, j + 1] = dp[i, j]
                bt[i, j + 1] = 3

    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 1:
            i -= 1
            j -= 1
            pairs.append((i, j))
        elif move == 2:
            i -= 1
            pairs.append((i, None))
        elif move == 3:
            j -= 1
            pairs.append((None, j))
        else:
            break
    pairs.reverse()
    return pairs

# --- Cell 4: Entry Point & Example Usage -------------------------------------
if __name__ == "__main__":
    # Run with hardcoded paths
    main(
        input_path=Path("C:/Users/junto/Downloads/head-repo/private725/PC2024/split_root/input_p.xlsx"),
        output_path=Path("C:/Users/junto/Downloads/head-repo/private725/PC2024/split_root/output_p.xlsx"),
        batch_size=64,
        max_workers=DEFAULT_MAX_WORKERS,
        device="cuda" if torch.cuda.is_available() else "cpu"