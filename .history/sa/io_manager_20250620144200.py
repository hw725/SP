"""개선된 병렬 처리 - 작업 단위 분산"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, Any, List
import math

logger = logging.getLogger(__name__)

# 전역 변수 (각 프로세스에서 초기화)
worker_embed_func = None
worker_modules = {}

def init_worker():
    """워커 프로세스 초기화 - 한 번만 실행"""
    global worker_embed_func, worker_modules
    try:
        print(f"워커 {mp.current_process().pid}: 초기화 시작")
        
        # 임베더 초기화
        from embedder import get_embed_func
        worker_embed_func = get_embed_func()
        
        # 필요한 모듈들 임포트
        from tokenizer import split_src_meaning_units, split_tgt_by_src_units_semantic
        from aligner import align_src_tgt
        from punctuation import mask_brackets, restore_brackets
        
        worker_modules = {
            'split_src_meaning_units': split_src_meaning_units,
            'split_tgt_by_src_units_semantic': split_tgt_by_src_units_semantic,
            'align_src_tgt': align_src_tgt,
            'mask_brackets': mask_brackets,
            'restore_brackets': restore_brackets
        }
        
        print(f"워커 {mp.current_process().pid}: 초기화 완료")
        
    except Exception as e:
        print(f"워커 {mp.current_process().pid}: 초기화 실패: {e}")
        worker_embed_func = None
        worker_modules = {}

def process_batch_sentences(sentence_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """배치 단위 문장 처리 - 한 프로세스가 여러 문장 처리"""
    global worker_embed_func, worker_modules
    
    if worker_embed_func is None or not worker_modules:
        print(f"워커 {mp.current_process().pid}: 초기화되지 않음")
        return []
    
    results = []
    
    for sentence_data in sentence_batch:
        try:
            sentence_id = sentence_data['sentence_id']
            src_text = sentence_data['src_text']
            tgt_text = sentence_data['tgt_text']
            
            # 모듈들 가져오기
            mask_brackets = worker_modules['mask_brackets']
            restore_brackets = worker_modules['restore_brackets']
            split_src_meaning_units = worker_modules['split_src_meaning_units']
            split_tgt_by_src_units_semantic = worker_modules['split_tgt_by_src_units_semantic']
            align_src_tgt = worker_modules['align_src_tgt']
            
            # 처리 파이프라인
            masked_src, src_masks = mask_brackets(src_text, 'source')
            masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
            
            src_units = split_src_meaning_units(masked_src)
            tgt_units = split_tgt_by_src_units_semantic(
                src_units, masked_tgt, worker_embed_func, min_tokens=1
            )
            aligned_pairs = align_src_tgt(src_units, tgt_units, worker_embed_func)
            
            # 결과 생성
            for phrase_idx, (src_unit, tgt_unit) in enumerate(aligned_pairs, 1):
                restored_src = restore_brackets(src_unit, src_masks)
                restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                
                results.append({
                    '문장식별자': sentence_id,
                    '구식별자': phrase_idx,
                    '원문구': restored_src,
                    '번역구': restored_tgt
                })
            
        except Exception as e:
            print(f"워커 {mp.current_process().pid}: 문장 {sentence_data.get('sentence_id', '?')} 실패: {e}")
            continue
    
    print(f"워커 {mp.current_process().pid}: 배치 {len(sentence_batch)}개 문장 처리 완료 → {len(results)}개 구")
    return results

def process_file(input_path: str, output_path: str, parallel: bool = False, workers: int = 4, batch_size: int = 20):
    """개선된 병렬 처리"""
    
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
    
    logger.info(f"로드 완료: {len(df)}개 행")
    
    # 필수 컬럼 확인
    required_columns = ['문장식별자', '원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
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
    
    if parallel and len(sentence_data_list) > workers:
        logger.info(f"배치 병렬 처리 시작 ({workers}개 프로세스)")
        
        # 문장들을 배치로 분할
        batch_size_per_worker = max(1, len(sentence_data_list) // workers)
        sentence_batches = []
        
        for i in range(0, len(sentence_data_list), batch_size_per_worker):
            batch = sentence_data_list[i:i + batch_size_per_worker]
            sentence_batches.append(batch)
        
        logger.info(f"배치 분할: {len(sentence_batches)}개 배치, 배치당 평균 {batch_size_per_worker}개 문장")
        
        # 프로세스 풀로 배치 처리
        with mp.Pool(processes=workers, initializer=init_worker) as pool:
            try:
                # 비동기 배치 처리
                async_results = []
                for i, batch in enumerate(sentence_batches):
                    async_result = pool.apply_async(process_batch_sentences, (batch,))
                    async_results.append((i, async_result))
                
                # 결과 수집
                for batch_idx, async_result in tqdm(async_results, desc="배치 처리"):
                    try:
                        batch_results = async_result.get(timeout=300)  # 5분 타임아웃
                        results.extend(batch_results)
                        logger.info(f"배치 {batch_idx+1}/{len(sentence_batches)} 완료: {len(batch_results)}개 구")
                    except mp.TimeoutError:
                        logger.error(f"배치 {batch_idx+1} 타임아웃")
                    except Exception as e:
                        logger.error(f"배치 {batch_idx+1} 처리 오류: {e}")
                        
            except KeyboardInterrupt:
                logger.info("사용자 중단")
                pool.terminate()
                pool.join()
                raise
                
    else:
        logger.info("순차 처리 시작")
        
        # 순차 처리 (기존 코드와 동일)
        from embedder import get_embed_func
        embed_func = get_embed_func()
        
        for sentence_data in tqdm(sentence_data_list, desc="문장 처리"):
            # ... 기존 순차 처리 로직 ...
            pass
    
    # 결과 저장 (기존과 동일)
    if not results:
        raise ValueError("처리된 결과가 없습니다")
    
    result_df = pd.DataFrame(results)
    
    if output_path.endswith('.xlsx'):
        result_df.to_excel(output_path, index=False)
    else:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"처리 완료: {len(sentence_data_list)}개 문장 → {len(results)}개 구")
    logger.info(f"출력: {output_path}")