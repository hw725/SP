# Standard library imports
import argparse
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
import torch
from packaging import version
from sklearn.metrics.pairwise import cosine_similarity

# 임시 해결책: 내장 함수로 대체
def load_excel(file_path: Path) -> pd.DataFrame:
    """Load Excel file using pandas."""
    return pd.read_excel(file_path)

def write_excel(file_path: Path, data: pd.DataFrame) -> None:
    """Write DataFrame to Excel file."""
    data.to_excel(file_path, index=False)

# 임시 해결책: 더미 함수들
def split_src_meaning_units(texts: List[str], max_workers: int = 4) -> List[List[str]]:
    """Dummy function for source text splitting."""
    # 간단한 문장 분할
    result = []
    for text in texts:
        sentences = text.split('. ')
        result.append([s.strip() for s in sentences if s.strip()])
    return result

def split_tgt_meaning_units(texts: List[str], max_workers: int = 4) -> List[List[str]]:
    """Dummy function for target text splitting."""
    # 간단한 문장 분할
    result = []
    for text in texts:
        sentences = text.split('. ')
        result.append([s.strip() for s in sentences if s.strip()])
    return result

def compute_embeddings(texts: List[str], batch_size: int = 32, device: str = 'cpu') -> np.ndarray:
    """Dummy function for embedding computation using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if not texts:
        return np.array([]).reshape(0, 100)  # Empty array with fixed dimension
    
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    try:
        embeddings = vectorizer.fit_transform(texts).toarray()
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings
    except Exception:
        # Fallback to random embeddings
        return np.random.randn(len(texts), 100)

def align_pairs(src_units: List[List[str]], tgt_units: List[List[str]], 
                src_emb: List[np.ndarray], tgt_emb: List[np.ndarray]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Simple alignment using cosine similarity."""
    alignments = []
    
    for doc_idx in range(len(src_units)):
        if not src_units[doc_idx] or not tgt_units[doc_idx]:
            continue
            
        src_doc_emb = src_emb[doc_idx]
        tgt_doc_emb = tgt_emb[doc_idx]
        
        if src_doc_emb.size == 0 or tgt_doc_emb.size == 0:
            continue
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(tgt_doc_emb, src_doc_emb)
        
        # Simple greedy alignment
        for tgt_idx in range(len(tgt_units[doc_idx])):
            src_idx = np.argmax(sim_matrix[tgt_idx])
            alignments.append((src_idx, tgt_idx))
    
    return alignments

def align_by_similarity(src_units: List[List[str]], tgt_units: List[List[str]], 
                       model: Any, sim_threshold: float = 0.35) -> List[Tuple[Optional[int], Optional[int]]]:
    """Similarity-based alignment."""
    # Simplified version of similarity alignment
    return align_pairs(src_units, tgt_units, [], [])  # Will be handled by calling function

# --- Configuration & Constants ---------------------------------------------
REQUIRED_LIBS = {
    'pandas': '1.3.0',
    'numpy': '1.21.0',
    'sklearn': '0.24.0',
    'packaging': '21.0',
}
DEFAULT_MAX_WORKERS = os.cpu_count() or 4

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Dependency Check -------------------------------------------------------
def check_dependencies(skip: bool = False) -> None:
    """Check if required dependencies are installed with correct versions."""
    if skip:
        logger.info("Skipping dependency check")
        return
    
    missing = []
    for lib, min_ver in REQUIRED_LIBS.items():
        try:
            if lib == 'sklearn':
                module = __import__('sklearn')
            else:
                module = __import__(lib)
            inst_ver = getattr(module, '__version__', '0.0.0')
            if version.parse(inst_ver) < version.parse(min_ver):
                missing.append(f"{lib}>={min_ver} (installed: {inst_ver})")
        except ImportError:
            missing.append(f"{lib}>={min_ver} (not installed)")
    
    if missing:
        logger.warning(f'Some dependencies may be outdated: {"; ".join(missing)}')
    else:
        logger.info("All dependencies check passed")

# --- Pipeline Steps --------------------------------------------------------
def load_data(input_path: Path) -> pd.DataFrame:
    """Load Excel data and validate required columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        df = load_excel(input_path)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error("Failed to load Excel: %s", e)
        raise
    
    if df.empty:
        raise ValueError("Input file is empty")
    
    required = {'원문', '번역문'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Remove rows with NaN values in required columns
    initial_count = len(df)
    df = df.dropna(subset=['원문', '번역문'])
    final_count = len(df)
    if final_count < initial_count:
        logger.warning(f"Dropped {initial_count - final_count} rows with missing values")
    
    if df.empty:
        raise ValueError("No valid rows remaining after removing NaN values")
    
    return df

def split_units(texts: List[str], is_source: bool, workers: int) -> List[List[str]]:
    """Split texts into meaning units."""
    if not texts:
        raise ValueError("Empty text list provided")
    
    fn = split_src_meaning_units if is_source else split_tgt_meaning_units
    text_type = 'source' if is_source else 'target'
    logger.info(f"Splitting {text_type} text into meaning units (# texts={len(texts)})...")
    
    try:
        units = fn(texts, max_workers=workers)
        total_units = sum(len(unit_list) for unit_list in units)
        logger.info(f"Split into {total_units} {text_type} meaning units")
        return units
    except Exception as e:
        logger.error("Tokenization error: %s", e)
        raise

def embed_units(units: List[List[str]], batch_size: int, device: str) -> List[np.ndarray]:
    """Compute embeddings for meaning units."""
    if not units:
        raise ValueError("Empty units list provided")
    
    logger.info(f"Computing embeddings for {len(units)} documents (batch={batch_size}, device={device})...")
    
    try:
        embeddings = []
        total_units = 0
        
        for i, unit_list in enumerate(units):
            if not unit_list:
                logger.warning(f"Document {i} has no meaning units, using zero embeddings")
                embeddings.append(np.array([]).reshape(0, 100))
                continue
                
            # Compute embeddings for this document's units
            doc_emb = compute_embeddings(unit_list, batch_size=batch_size, device=device)
            embeddings.append(doc_emb)
            total_units += len(unit_list)
            
        logger.info(f"Computed embeddings for {total_units} total units across {len(units)} documents")
        return embeddings
    except Exception as e:
        logger.error("Embedding computation failed: %s", e)
        raise

def align_units(
    src_units: List[List[str]],
    tgt_units: List[List[str]],
    src_emb: List[np.ndarray],
    tgt_emb: List[np.ndarray],
    method: str = 'dp',
    **kwargs
) -> List[Dict[str, Any]]:
    """Align source and target meaning units."""
    if not src_units or not tgt_units:
        raise ValueError("Empty units provided for alignment")
    
    if len(src_units) != len(tgt_units):
        raise ValueError(f"Mismatch in document count: src={len(src_units)}, tgt={len(tgt_units)}")
    
    all_alignments = []
    
    for doc_idx, (src_doc_units, tgt_doc_units, src_doc_emb, tgt_doc_emb) in enumerate(
        zip(src_units, tgt_units, src_emb, tgt_emb)
    ):
        logger.info(f"Aligning document {doc_idx + 1}/{len(src_units)}")
        
        if not src_doc_units or not tgt_doc_units:
            logger.warning(f"Document {doc_idx} has empty units, skipping alignment")
            continue
        
        if method == 'dp':
            try:
                # Simple greedy alignment using cosine similarity
                if src_doc_emb.size == 0 or tgt_doc_emb.size == 0:
                    continue
                    
                sim_matrix = cosine_similarity(tgt_doc_emb, src_doc_emb)
                
                for tgt_idx in range(len(tgt_doc_units)):
                    src_idx = np.argmax(sim_matrix[tgt_idx])
                    
                    alignment = {
                        'doc_idx': doc_idx,
                        'src_unit_idx': src_idx,
                        'tgt_unit_idx': tgt_idx,
                        'src_unit': src_doc_units[src_idx],
                        'tgt_unit': tgt_doc_units[tgt_idx],
                        'similarity': sim_matrix[tgt_idx, src_idx],
                        'method': 'dp'
                    }
                    all_alignments.append(alignment)
                    
            except Exception as e:
                logger.error(f"DP alignment failed for document {doc_idx}: %s", e)
                continue
        else:
            raise ValueError(f"Unknown alignment method: {method}")
    
    logger.info(f"Generated {len(all_alignments)} total alignment pairs")
    return all_alignments

def prepare_output_data(
    df: pd.DataFrame,
    alignments: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Prepare output data with alignment results."""
    output_rows = []
    
    for i, alignment in enumerate(alignments):
        doc_idx = alignment['doc_idx']
        
        row = {
            'alignment_id': i,
            'doc_idx': doc_idx,
            'src_unit_idx': alignment['src_unit_idx'],
            'tgt_unit_idx': alignment['tgt_unit_idx'],
            'src_unit': alignment['src_unit'],
            'tgt_unit': alignment['tgt_unit'],
            'similarity': alignment.get('similarity', 0.0),
            'src_text': df.iloc[doc_idx]['원문'] if doc_idx < len(df) else None,
            'tgt_text': df.iloc[doc_idx]['번역문'] if doc_idx < len(df) else None,
            'method': alignment['method']
        }
        output_rows.append(row)
    
    return pd.DataFrame(output_rows)

# --- CLI & Main -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Meaning-unit alignment pipeline"
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Input .xlsx path')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output .xlsx path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Embedding batch size')
    parser.add_argument('--max-workers', type=int,
                        default=DEFAULT_MAX_WORKERS,
                        help='Thread count for tokenization')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Torch device')
    parser.add_argument('--method', choices=['dp', 'sim'], default='dp',
                        help='Alignment method')
    parser.add_argument('--sim-threshold', type=float, default=0.35,
                        help='Similarity threshold for sim method')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip dependency version check')
    return parser.parse_args()

def main(parsed_args_override: Optional[argparse.Namespace] = None) -> None:
    """Execute the full pipeline."""
    try:
        # Use provided args or parse from command line
        if parsed_args_override is not None:
            args = parsed_args_override
        else:
            args = parse_args()
        
        # Check dependencies unless skipped
        check_dependencies(skip=getattr(args, 'skip_deps', False))
        
        logger.info("Starting meaning-unit alignment pipeline...")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Method: {args.method}")
        
        # Load and validate data
        df = load_data(args.input)
        src_texts = df['원문'].astype(str).tolist()
        tgt_texts = df['번역문'].astype(str).tolist()
        
        # Split into meaning units
        src_units = split_units(src_texts, is_source=True, workers=args.max_workers)
        tgt_units = split_units(tgt_texts, is_source=False, workers=args.max_workers)
        
        # Compute embeddings (per document)
        src_emb = embed_units(src_units, args.batch_size, args.device)
        tgt_emb = embed_units(tgt_units, args.batch_size, args.device)
        
        # Align units
        alignments = align_units(
            src_units, tgt_units, src_emb, tgt_emb,
            method=args.method,
            sim_threshold=getattr(args, 'sim_threshold', 0.35),
            model=None
        )
        
        # Prepare output data
        output_df = prepare_output_data(df, alignments)
        
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output
        logger.info(f"Writing output to {args.output}")
        write_excel(args.output, output_df)
            
        logger.info(f"Pipeline completed successfully. Output contains {len(output_df)} alignment pairs.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    # Run with hardcoded arguments for testing
    hardcoded_args = argparse.Namespace(
        input=Path("C:/Users/junto/Downloads/head-repo/SP/split_p/input_p.xlsx"),
        output=Path("C:/Users/junto/Downloads/head-repo/SP/split_p/output_p.xlsx"),
        batch_size=64,
        max_workers=DEFAULT_MAX_WORKERS,
        device="cpu",  # torch 문제로 인해 cpu 사용
        method="dp",
        sim_threshold=0.35,
        skip_deps=True  # 의존성 체크 건너뛰기
    )
    
    main(parsed_args_override=hardcoded_args)
