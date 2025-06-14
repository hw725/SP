# io_manager.py - 문장을 구 단위로 정렬하는 I/O 처리 모듈

import os
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from tokenizer import split_src_meaning_units, split_tgt_meaning_units
from punctuation import mask_brackets, restore_masks
import aligner

try:
    from embedder import compute_embeddings_with_cache
except ImportError:
    from embedder import compute_embeddings_with_cache

logger = logging.getLogger(__name__)

def process_file(
    input_path: str,
    output_path: str,
    batch_size: int = 128,
    verbose: bool = False
) -> None:
    """
    Excel 파일을 읽어 원문-번역문 병렬 정렬을 수행하고 결과를 저장.
    """

    try:
        df = pd.read_excel(input_path, engine='openpyxl')
    except Exception as e:
        logger.error(f"[IO] Excel 파일 읽기 실패: {e}")
        print(f"[IO] Excel 파일 읽기 실패: {e}")
        return

    if '원문' not in df.columns or '번역문' not in df.columns:
        logger.error("[IO] '원문' 또는 '번역문' 칼럼이 없습니다.")
        print("[IO] '원문' 또는 '번역문' 칼럼이 없습니다.")
        return

    MAX_M = 0
    for text in df['번역문'].fillna(""):
        try:
            units = split_tgt_meaning_units(
                "",
                str(text),
                use_semantic=False
            )
            MAX_M = max(MAX_M, len(units))
        except Exception as e:
            logger.warning(f"[IO] 번역문 의미 단위 길이 계산 중 오류: {e}")

    aligner.dp_prev_global = np.ones(MAX_M + 1) * -np.inf
    aligner.dp_curr_global = np.ones(MAX_M + 1) * -np.inf

    outputs: List[Dict[str, Any]] = []
    total_rows = len(df)
    iterator = tqdm(df.itertuples(index=False), total=total_rows, desc="Processing rows")

    for idx, row in enumerate(iterator, start=1):
        src_text = str(getattr(row, '원문', '') or '')
        tgt_text = str(getattr(row, '번역문', '') or '')
        print(f"\n[========= ROW {idx} =========]")
        print("원문(입력):", src_text)
        print("번역문(입력):", tgt_text)

        try:
            masked_src, src_masks = mask_brackets(src_text, text_type="source")
            masked_tgt, tgt_masks = mask_brackets(tgt_text, text_type="target")

            tgt_units = split_tgt_meaning_units(
                masked_src,
                masked_tgt,
                use_semantic=True,
                min_tokens=1,
                max_tokens=50
            )
            tgt_units = [restore_masks(unit, tgt_masks) for unit in tgt_units]
            src_units = split_src_meaning_units(masked_src)

            aligned_pairs = aligner.align_src_tgt(
                src_units,
                tgt_units,
                compute_embeddings_with_cache
            )
            aligned_src_units, aligned_tgt_units = zip(*aligned_pairs)
            aligned_src_units = [restore_masks(unit, src_masks) for unit in aligned_src_units]
            aligned_tgt_units = [restore_masks(unit, tgt_masks) for unit in aligned_tgt_units]

            # 구 단위 매칭 결과 (디버그 or 확인용)
            print("정렬 결과: (구 대 구 대조)")
            for src_gu, tgt_gu in zip(aligned_src_units, aligned_tgt_units):
                print(f"SRC: {src_gu} | TGT: {tgt_gu}")

            # --- 핵심 변경 부분 ---
            for gu_idx, (src_gu, tgt_gu) in enumerate(zip(aligned_src_units, aligned_tgt_units), start=1):
                outputs.append({
                    "문장식별자": idx,
                    "구식별자": gu_idx,
                    "원문구": src_gu,
                    "번역구": tgt_gu
                })

            if verbose:
                logger.info(f"[IO] 행 {idx} 정렬 완료")

        except Exception as e:
            logger.warning(f"[IO] 행 {idx} 처리 실패: {e}")
            outputs.append({
                "문장식별자": idx,
                "구식별자": 1,
                "원문구": src_text,
                "번역구": tgt_text
            })

    try:
        output_df = pd.DataFrame(outputs, columns=["문장식별자", "구식별자", "원문구", "번역구"])
        output_df.to_excel(output_path, index=False, engine='openpyxl')
        if verbose:
            logger.info(f"[IO] 결과 저장 완료: {output_path}")
    except Exception as e:
        logger.error(f"[IO] 결과 저장 실패: {e}")
        print(f"[IO] 결과 저장 실패: {e}")