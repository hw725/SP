"""파일 처리 메인 로직 - parallel 옵션 추가"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def process_single_sentence(sentence_data: Dict[str, Any], embed_func, batch_size: int = 20) -> List[Dict[str, Any]]:
    """단일 문장 처리"""
    from tokenizer import split_src_meaning_units, split_tgt_by_src_units_semantic
    from aligner import align_src_tgt
    from punctuation import mask_brackets, restore_brackets
    
    try:
        sentence_id = sentence_data['sentence_id']
        src_text = sentence_data['src_text']
        tgt_text = sentence_data['tgt_text']
        
        # 1. 괄호 마스킹
        masked_src, src_masks = mask_brackets(src_text, 'source')
        masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
        
        # 2. 원문 분할
        src_units = split_src_meaning_units(masked_src)
        
        # 3. 번역문 의미 기반 분할
        tgt_units = split_tgt_by_src_units_semantic(
            src_units, masked_tgt, embed_func, min_tokens=1
        )
        
        # 4. 정렬
        aligned_pairs = align_src_tgt(src_units, tgt_units, embed_func)
        
        # 5. 결과 생성
        results = []
        for phrase_idx, (src_unit, tgt_unit) in enumerate(aligned_pairs, 1):
            restored_src = restore_brackets(src_unit, src_masks)
            restored_tgt = restore_brackets(tgt_unit, tgt_masks)
            
            results.append({
                '문장식별자': sentence_id,
                '구식별자': phrase_idx,
                '원문구': restored_src,
                '번역구': restored_tgt
            })
        
        return results
        
    except Exception as e:
        logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 처리 실패: {e}")
        return []

def process_file(input_path: str, output_path: str, parallel: bool = False, workers: int = 4, batch_size: int = 20):
    """파일 처리 - 병렬 옵션 추가"""
    
    # 동적 임포트
    from embedder import get_embed_func
    
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
    
    # 데이터 준비
    sentence_data_list = []
    for idx, row in df.iterrows():
        src_text = str(row['원문']).strip()
        tgt_text = str(row['번역문']).strip()
        
        if src_text and tgt_text:
            sentence_data_list.append({
                'sentence_id': row['문장식별자'],
                'src_text': src_text,
                'tgt_text': tgt_text
            })
    
    logger.info(f"처리할 문장: {len(sentence_data_list)}개")
    
    # 처리 실행
    results = []
    
    if parallel and len(sentence_data_list) > 1:
        logger.info(f"병렬 처리 시작 ({workers}개 워커)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for sentence_data in sentence_data_list:
                future = executor.submit(process_single_sentence, sentence_data, embed_func, batch_size)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="문장 처리"):
                try:
                    sentence_results = future.result()
                    results.extend(sentence_results)
                except Exception as e:
                    logger.error(f"병렬 처리 오류: {e}")
    else:
        logger.info("순차 처리 시작")
        for sentence_data in tqdm(sentence_data_list, desc="문장 처리"):
            sentence_results = process_single_sentence(sentence_data, embed_func, batch_size)
            results.extend(sentence_results)
    
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
    logger.info(f"총 문장: {len(sentence_data_list)}개")
    logger.info(f"총 구: {len(results)}개")
    logger.info(f"평균 구/문장: {len(results)/len(sentence_data_list):.1f}")
    logger.info(f"출력 파일: {output_path}")
    logger.info(f"출력 컬럼: {list(result_df.columns)}")