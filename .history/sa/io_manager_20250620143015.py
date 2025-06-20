"""파일 처리 메인 로직 - 진짜 병렬 처리"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, Any, List
import pickle
import os

logger = logging.getLogger(__name__)

def init_worker():
    """워커 프로세스 초기화"""
    global worker_embed_func
    try:
        # 각 워커에서 독립적으로 임베더 초기화
        from embedder import get_embed_func
        worker_embed_func = get_embed_func()
        print(f"워커 {os.getpid()}: 임베더 초기화 완료")
    except Exception as e:
        print(f"워커 {os.getpid()}: 임베더 초기화 실패: {e}")
        worker_embed_func = None

def process_single_sentence_worker(sentence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """워커 프로세스에서 실행되는 문장 처리 함수"""
    try:
        global worker_embed_func
        if worker_embed_func is None:
            return []
        
        # 필요한 모듈들 import
        from tokenizer import split_src_meaning_units, split_tgt_by_src_units_semantic
        from aligner import align_src_tgt
        from punctuation import mask_brackets, restore_brackets
        
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
            src_units, masked_tgt, worker_embed_func, min_tokens=1
        )
        
        # 4. 정렬
        aligned_pairs = align_src_tgt(src_units, tgt_units, worker_embed_func)
        
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
        
        print(f"워커 {os.getpid()}: 문장 {sentence_id} 처리 완료 ({len(results)}개 구)")
        return results
        
    except Exception as e:
        print(f"워커 {os.getpid()}: 문장 {sentence_data.get('sentence_id', '?')} 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_file(input_path: str, output_path: str, parallel: bool = False, workers: int = 4, batch_size: int = 20):
    """파일 처리 - 진짜 병렬 처리"""
    
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
        logger.info(f"병렬 처리 시작 ({workers}개 프로세스)")
        
        # 프로세스 풀 사용
        with mp.Pool(processes=workers, initializer=init_worker) as pool:
            try:
                # 비동기 작업 제출
                async_results = []
                for sentence_data in sentence_data_list:
                    async_result = pool.apply_async(process_single_sentence_worker, (sentence_data,))
                    async_results.append(async_result)
                
                # 결과 수집 (진행률 표시)
                for i, async_result in enumerate(tqdm(async_results, desc="문장 처리")):
                    try:
                        sentence_results = async_result.get(timeout=120)  # 2분 타임아웃
                        results.extend(sentence_results)
                    except mp.TimeoutError:
                        logger.error(f"문장 {sentence_data_list[i]['sentence_id']} 처리 타임아웃")
                    except Exception as e:
                        logger.error(f"문장 {sentence_data_list[i]['sentence_id']} 처리 오류: {e}")
                        
            except KeyboardInterrupt:
                logger.info("사용자 중단 요청")
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                logger.error(f"병렬 처리 오류: {e}")
                pool.terminate()
                pool.join()
                raise
    else:
        logger.info("순차 처리 시작")
        
        # 순차 처리용 임베더 초기화
        from embedder import get_embed_func
        embed_func = get_embed_func()
        
        for sentence_data in tqdm(sentence_data_list, desc="문장 처리"):
            try:
                from tokenizer import split_src_meaning_units, split_tgt_by_src_units_semantic
                from aligner import align_src_tgt
                from punctuation import mask_brackets, restore_brackets
                
                sentence_id = sentence_data['sentence_id']
                src_text = sentence_data['src_text']
                tgt_text = sentence_data['tgt_text']
                
                # 처리 로직
                masked_src, src_masks = mask_brackets(src_text, 'source')
                masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
                
                src_units = split_src_meaning_units(masked_src)
                tgt_units = split_tgt_by_src_units_semantic(src_units, masked_tgt, embed_func, min_tokens=1)
                aligned_pairs = align_src_tgt(src_units, tgt_units, embed_func)
                
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
                logger.error(f"문장 {sentence_data['sentence_id']} 순차 처리 실패: {e}")
                continue
    
    # 결과 저장
    logger.info(f"결과 저장: {output_path}")
    
    if not results:
        logger.error("처리된 결과가 없습니다")
        raise ValueError("처리된 결과가 없습니다")
    
    result_df = pd.DataFrame(results)
    
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