"""파일 처리 메인 로직 - 원본 기반"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def process_file(input_path: str, output_path: str):
    """파일 처리 - 구 단위 출력"""
    
    # 동적 임포트 (교체된 파일들 사용)
    from tokenizer import split_src_meaning_units, split_tgt_by_src_units_semantic
    from aligner import align_src_tgt
    from embedder import get_embed_func
    from punctuation import mask_brackets, restore_brackets
    
    # 데이터 로드
    logger.info(f"파일 로드: {input_path}")
    
    try:
        if input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"파일 로드 실패: {e}")
        raise
    
    logger.info(f"로드 완료: {len(df)}개 행, 컬럼: {list(df.columns)}")
    
    # 필수 컬럼 확인
    required_columns = ['문장식별자', '원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"필수 컬럼 누락: {missing_columns}")
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
    # 임베딩 함수 초기화
    logger.info("임베딩 함수 초기화...")
    embed_func = get_embed_func()
    logger.info("임베딩 함수 초기화 완료")
    
    # 결과 리스트
    results = []
    total_phrases = 0
    
    # 각 행 처리
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="문장 처리"):
        try:
            sentence_id = row['문장식별자']
            src_text = str(row['원문']).strip()
            tgt_text = str(row['번역문']).strip()
            
            if not src_text or not tgt_text:
                logger.warning(f"행 {idx}: 빈 텍스트 스킵")
                continue
            
            logger.debug(f"행 {idx} 처리: 문장ID={sentence_id}")
            logger.debug(f"  원문: {src_text[:50]}...")
            logger.debug(f"  번역문: {tgt_text[:50]}...")
            
            # 1. 괄호 마스킹
            masked_src, src_masks = mask_brackets(src_text, 'source')
            masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
            
            # 2. 원문 분할 (jieba 고정)
            src_units = split_src_meaning_units(masked_src)
            logger.debug(f"  원문 {len(src_units)}개 단위로 분할")
            
            # 3. 번역문 의미 기반 분할
            tgt_units = split_tgt_by_src_units_semantic(
                src_units, masked_tgt, embed_func, min_tokens=1
            )
            logger.debug(f"  번역문 {len(tgt_units)}개 단위로 분할")
            
            # 4. 정렬
            aligned_pairs = align_src_tgt(src_units, tgt_units, embed_func)
            logger.debug(f"  {len(aligned_pairs)}개 쌍으로 정렬")
            
            # 5. 결과 생성 (구 단위)
            for phrase_idx, (src_unit, tgt_unit) in enumerate(aligned_pairs, 1):
                # 괄호 복원
                restored_src = restore_brackets(src_unit, src_masks)
                restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                
                results.append({
                    '문장식별자': sentence_id,
                    '구식별자': phrase_idx,
                    '원문구': restored_src,
                    '번역구': restored_tgt
                })
                
            total_phrases += len(aligned_pairs)
            
        except Exception as e:
            logger.error(f"행 {idx} 처리 실패: {e}")
            continue
    
    # 결과 저장
    logger.info(f"결과 저장: {output_path}")
    result_df = pd.DataFrame(results)
    
    if result_df.empty:
        logger.error("처리된 결과가 없습니다")
        raise ValueError("처리된 결과가 없습니다")
    
    try:
        if output_path.endswith('.xlsx'):
            result_df.to_excel(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")
        raise
    
    logger.info("=== 처리 결과 ===")
    logger.info(f"총 문장: {len(df)}개")
    logger.info(f"총 구: {total_phrases}개")
    logger.info(f"평균 구/문장: {total_phrases/len(df):.1f}")
    logger.info(f"출력 파일: {output_path}")
    logger.info(f"출력 컬럼: {list(result_df.columns)}")