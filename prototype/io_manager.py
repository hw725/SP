# io_manager.py - 문장을 구 단위로 정렬하는 I/O 처리 모듈

import os
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# tokenizer, embedder는 기존대로 기능만 가져옵니다.
from tokenizer import split_src_meaning_units, split_tgt_meaning_units
from embedder import compute_embeddings_with_cache

# 변경: from aligner import ... 대신, aligner 전체를 import
import aligner

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def process_excel(
    input_path: str,
    output_path: str,
    batch_size: int = 128,
    verbose: bool = False
) -> None:
    """
    1) input_path(.xls/.xlsx) 읽어서
    2) 각 행(row)에 대해 토큰 분할 → 정렬 → 결과 누적
    3) 최종 DataFrame으로 만들어 output_path(.xlsx)로 저장
    """
    # verbose 옵션에 따라 로그 레벨 조정
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # 1) 엑셀 읽기
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
    except Exception as e:
        logger.error(f"[IO] Excel 파일 읽기 실패: {e}")
        return

    # 2) 필수 컬럼 존재 확인
    if '원문' not in df.columns or '번역문' not in df.columns:
        logger.error("[IO] '원문' 또는 '번역문' 칼럼이 없습니다.")
        return

    # 3) 번역문 열 전체 훑어서 MAX_M(최대 토큰 길이) 계산
    MAX_M = 0
    for text in df['번역문'].fillna(""):
        tk = split_tgt_meaning_units(str(text))
        if len(tk) > MAX_M:
            MAX_M = len(tk)

    # 4) 전역 DP 배열 초기화 (크기: MAX_M + 1)
    #    → 반드시 aligner 모듈의 전역 변수에 직접 대입해야 함
    aligner.dp_prev_global = np.ones(MAX_M + 1) * -np.inf
    aligner.dp_curr_global = np.ones(MAX_M + 1) * -np.inf

    outputs: List[Dict[str, Any]] = []
    total_rows = len(df)

    # 5) 행(row) 단위 처리
    for idx, row in enumerate(
        tqdm(df.itertuples(index=False), total=total_rows, desc="Processing rows"), 
        start=1
    ):
        # 5.1) 원문/번역문 텍스트 추출 (NaN 처리)
        src_text = "" if pd.isna(getattr(row, '원문')) else str(getattr(row, '원문'))
        tgt_text = "" if pd.isna(getattr(row, '번역문')) else str(getattr(row, '번역문'))

        # 5.2) 토큰 분할
        try:
            src_units = split_src_meaning_units(src_text)
        except Exception as e:
            logger.error(f"[IO] 행 {idx} 원문 분할 오류: {e}")
            src_units = []

        try:
            tgt_units = split_tgt_meaning_units(tgt_text)
        except Exception as e:
            logger.error(f"[IO] 행 {idx} 번역문 분할 오류: {e}")
            tgt_units = []

        # 5.3) 정렬 (aligner.align_src_tgt 호출)
        try:
            aligned = aligner.align_src_tgt(
                src_units,
                tgt_units,
                compute_embeddings_with_cache,
                batch_size=batch_size,
                show_batch_progress=verbose
            )
        except Exception as e:
            logger.error(f"[IO] 행 {idx} 정렬 오류: {e}")
            aligned = [""] * len(src_units)

        # 5.4) 결과 누적 (문장식별자 = idx, 구식별자 = i)
        for i, (s_unit, t_seg) in enumerate(zip(src_units, aligned), start=1):
            outputs.append({
                '문장식별자': idx,
                '구식별자': str(i),
                '원문구': s_unit,
                '번역구': t_seg
            })

    # 6) 결과를 DataFrame으로 만들어 엑셀 저장
    df_out = pd.DataFrame(outputs)
    try:
        # 출력 경로의 디렉터리 부분만 꺼낸다. 비어 있으면 mkdirs 호출을 건너뛴다.
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out.to_excel(output_path, index=False, engine='openpyxl')
        print(f"[IO] 결과 저장 완료: {output_path}")
    except Exception as e:
        print(f"[IO] 결과 Excel 저장 실패: {e}")