"""개선된 병렬 처리 - 작업 단위 분산"""
try:
    import pandas as pd
except ImportError as e:
    import logging
    logging.error(f"\u274c pandas import 실패: {e}")
    pd = None
try:
    from tqdm import tqdm
except ImportError as e:
    import logging
    logging.error(f"\u274c tqdm import 실패: {e}")
    def tqdm(x, *args, **kwargs):
        return x
import logging
from pathlib import Path
import multiprocessing as mp
from typing import Dict, Any, List
import math
import sys, os

CSP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CSP_ROOT not in sys.path:
    sys.path.insert(0, CSP_ROOT)

logger = logging.getLogger(__name__)

# 전역 변수 (각 프로세스에서 초기화)
worker_embed_func = None
worker_modules = {}
verbose_mode = False  # 기본적으로 비활성화

def set_verbose_mode(verbose=False):
    """전역 verbose 모드 설정"""
    global verbose_mode
    verbose_mode = verbose

def init_worker(device_id=None, embedder_name='bge'):
    """워커 프로세스 초기화 - 한 번만 실행 (device_id로 GPU 분배, 임베더 선택)"""
    global worker_embed_func, worker_modules
    try:
        if verbose_mode:
            print(f"워커 {mp.current_process().pid}: 초기화 시작 (device_id={device_id}, embedder={embedder_name})")
        
        # 임베더 초기화 - 임베더 타입에 따라 다른 함수 사용
        if embedder_name == 'openai':
            from sa_embedders import get_embedder
            worker_embed_func = get_embedder('openai')
            if verbose_mode:
                print(f"워커 {mp.current_process().pid}: OpenAI 임베더 초기화 완료")
        else:  # bge 또는 기본값
            from sa_embedders import get_embed_func
            worker_embed_func = get_embed_func(device_id=device_id)
            if verbose_mode:
                print(f"워커 {mp.current_process().pid}: BGE 임베더 초기화 완료")
        
        # 필요한 모듈들 임포트
        from sa_tokenizers.jieba_mecab import split_src_meaning_units, split_tgt_meaning_units, split_tgt_by_src_units_semantic
        from punctuation import mask_brackets, restore_brackets
        
        worker_modules = {
            'split_src_meaning_units': split_src_meaning_units,
            'split_tgt_meaning_units': split_tgt_meaning_units,
            'split_tgt_by_src_units_semantic': split_tgt_by_src_units_semantic,
            'mask_brackets': mask_brackets,
            'restore_brackets': restore_brackets
        }
        
        if verbose_mode:
            print(f"워커 {mp.current_process().pid}: 초기화 완료 (device_id={device_id})")
        
    except Exception as e:
        # 초기화 실패는 중요한 오류이므로 항상 출력
        print(f"워커 {mp.current_process().pid}: 초기화 실패: {e}")
        worker_embed_func = None
        worker_modules = {}

def process_batch_sentences(sentence_batch: List[Dict[str, Any]], device_id=None) -> List[Dict[str, Any]]:
    """배치 단위 문장 처리 - 한 프로세스가 여러 문장 처리"""
    global worker_embed_func, worker_modules
    
    if worker_embed_func is None or not worker_modules:
        # 중요 오류는 항상 출력
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
            
            # 처리 파이프라인
            masked_src, src_masks = mask_brackets(src_text, 'source')
            masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
            
            src_units = split_src_meaning_units(masked_src)
            tgt_units = split_tgt_by_src_units_semantic(
                src_units, 
                masked_tgt, 
                worker_embed_func, 
                min_tokens=1
            )
            
            # 결과 생성
            for phrase_idx, (src_unit, tgt_unit) in enumerate(zip(src_units, tgt_units), 1):
                restored_src = restore_brackets(src_unit, src_masks)
                restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                
                results.append({
                    '문장식별자': sentence_id,
                    '구식별자': phrase_idx,
                    '원문': restored_src,
                    '번역문': restored_tgt
                })
            
        except Exception as e:
            if verbose_mode:
                print(f"워커 {mp.current_process().pid}: 문장 {sentence_data.get('sentence_id', '?')} 실패: {e}")
            logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 처리 오류: {e}")
            continue
    
    if verbose_mode:
        print(f"워커 {mp.current_process().pid}: 배치 {len(sentence_batch)}개 문장 처리 완료 → {len(results)}개 구")
    
    return results

def process_file(input_path: str, output_path: str, parallel: bool = False, workers: int = 4, 
                batch_size: int = 20, device_ids=None, verbose: bool = False, embedder_name: str = 'bge'):
    """개선된 병렬 처리 (임베더 선택 지원)"""
    
    # verbose 모드 설정
    set_verbose_mode(verbose)
    
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
        
        # device id 분배
        if device_ids is None:
            # 기본: 워커 수만큼 device id를 0,1,2,...로 분배 (단일 GPU면 모두 0)
            import torch
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpu > 1:
                device_ids = [i % n_gpu for i in range(workers)]
            else:
                device_ids = [0 for _ in range(workers)]
        
        # 프로세스 풀로 배치 처리
        logger.info("⏳ 워커 프로세스 초기화 중... (모델 로딩으로 인해 시간이 걸릴 수 있습니다)")
        print(f"⏳ {embedder_name.upper()} 모델 로딩 중... 잠시만 기다려주세요.")
        
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(device_ids[0], embedder_name)) as pool:
            try:
                async_results = []
                for i, batch in enumerate(sentence_batches):
                    device_id = device_ids[i % len(device_ids)]
                    async_result = pool.apply_async(process_batch_sentences, (batch,), {'device_id': device_id})
                    async_results.append((i, async_result))
                
                print(f"✅ 모델 로딩 완료! 문장 처리를 시작합니다...")
                logger.info("✅ 워커 프로세스 초기화 완료, 문장 처리 시작")
                
                # 결과 수집 - 문장 기준 진행률 표시
                processed_sentences = 0
                with tqdm(total=len(sentence_data_list), desc="문장 처리", unit="문장") as pbar:
                    for batch_idx, async_result in async_results:
                        try:
                            batch_results = async_result.get()  # 타임아웃 제거
                            results.extend(batch_results)
                            
                            # 현재 배치의 문장 수 계산
                            batch_sentences = len(sentence_batches[batch_idx])
                            processed_sentences += batch_sentences
                            pbar.update(batch_sentences)
                            pbar.set_postfix({"구": len(results), "배치": f"{batch_idx+1}/{len(sentence_batches)}"})
                            
                            logger.info(f"배치 {batch_idx+1}/{len(sentence_batches)} 완료: {batch_sentences}개 문장 → {len(batch_results)}개 구")
                        except Exception as e:
                            # 실패한 배치의 문장 수도 업데이트
                            batch_sentences = len(sentence_batches[batch_idx])
                            processed_sentences += batch_sentences
                            pbar.update(batch_sentences)
                            logger.error(f"배치 {batch_idx+1} 처리 오류: {e}")
                        
            except KeyboardInterrupt:
                logger.info("사용자 중단")
                pool.terminate()
                pool.join()
                raise
                
    else:
        logger.info("순차 처리 시작")
        
        # 순차 처리 - 임베더 선택
        print(f"⏳ {embedder_name.upper()} 모델 로딩 중... 잠시만 기다려주세요.")
        if embedder_name == 'openai':
            from sa_embedders import get_embedder
            embed_func = get_embedder('openai')
        else:  # bge 또는 기본값
            from sa_embedders import get_embed_func
            embed_func = get_embed_func()
        
        print(f"✅ 모델 로딩 완료! 문장 처리를 시작합니다...")
        logger.info("✅ 모델 로딩 완료, 순차 처리 시작")
        
        from sa_tokenizers.jieba_mecab import split_src_meaning_units, split_tgt_by_src_units_semantic
        from punctuation import mask_brackets, restore_brackets
        
        for sentence_data in tqdm(sentence_data_list, desc="문장 처리"):
            try:
                sentence_id = sentence_data['sentence_id']
                src_text = sentence_data['src_text']
                tgt_text = sentence_data['tgt_text']
                
                # 처리 파이프라인
                masked_src, src_masks = mask_brackets(src_text, 'source')
                masked_tgt, tgt_masks = mask_brackets(tgt_text, 'target')
                
                src_units = split_src_meaning_units(masked_src)
                tgt_units = split_tgt_by_src_units_semantic(
                    src_units, masked_tgt, embed_func, min_tokens=1
                )
                
                # 결과 생성
                for phrase_idx, (src_unit, tgt_unit) in enumerate(zip(src_units, tgt_units), 1):
                    restored_src = restore_brackets(src_unit, src_masks)
                    restored_tgt = restore_brackets(tgt_unit, tgt_masks)
                    
                    results.append({
                        '문장식별자': sentence_id,
                        '구식별자': phrase_idx,
                        '원문': restored_src,
                        '번역문': restored_tgt
                    })
                
            except Exception as e:
                logger.error(f"문장 {sentence_data.get('sentence_id', '?')} 순차 처리 오류: {e}")
                if verbose:
                    print(f"순차 처리 문장 {sentence_data.get('sentence_id', '?')} 실패: {e}")
                continue
    
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
    
    return result_df