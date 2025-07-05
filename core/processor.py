"""문장 단위 처리 및 정렬 모듈 - 진행률 표시 포함"""

import logging
import pandas as pd
import numpy as np
import time
import traceback  # 에러 추적용
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm  # 진행률 표시 추가
from core.io_utils import IOManager

try:
    from .tokenizers import split_src_meaning_units, split_tgt_meaning_units_sequential
except ImportError as e:
    import logging
    logging.error(f"sa_tokenizers import 실패: {e}")
    def split_src_meaning_units(*args, **kwargs):
        logging.error("토크나이저 기능을 사용할 수 없습니다.")
        return []
    def split_tgt_meaning_units(*args, **kwargs):
        logging.error("토크나이저 기능을 사용할 수 없습니다.")
        return []
try:
    from .embedders import compute_embeddings_with_cache
except ImportError as e:
    import logging
    logging.error(f"sa_embedders import 실패: {e}")
    def compute_embeddings_with_cache(*args, **kwargs):
        logging.error("임베더 기능을 사용할 수 없습니다.")
        import numpy as np
        return np.zeros((len(args[0]), 1024))  # fallback shape
try:
    from .alignment_functions import align_units
    from .embedders import get_embedder
except ImportError as e:
    import logging
    logging.error(f"필수 모듈 임포트 실패: {e}")
    def align_units(*args, **kwargs):
        logging.error("정렬 기능을 사용할 수 없습니다.")
        return []
    def get_embedder(*args, **kwargs):
        logging.error("임베딩 기능을 사용할 수 없습니다.")
        return None

# punctuation import 안전 처리
try:
    from .punctuation import process_punctuation
except ImportError:
    def process_punctuation(alignments, src_units, tgt_units):
        return alignments

logger = logging.getLogger(__name__)

def process_sentence(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    use_sequential: bool = False,  # 순차 모드 옵션 추가
    min_tokens: int = 1,
    max_tokens: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """단일 문장 처리"""
    print("[DEBUG] process_sentence 진입")
    try:
        print("[DEBUG] 1. 토크나이징 시작")
        src_units = split_src_meaning_units(
            src_text, 
            min_tokens=min_tokens, 
            max_tokens=max_tokens,
            use_advanced=True
        )
        if not src_units:
            logger.warning("Source units are empty after tokenization.")
            return {'src_units': [], 'tgt_units': [], 'alignments': [], 'status': 'failed', 'error': 'Empty source units'}

        print(f"[DEBUG] src_units: {src_units}")
        
        # Ensure embedder_name is passed to get_embedding_function
        embedder_name = kwargs.get("embedder_name", "bge")
        embed_func = get_embedding_function(embedder_name, **kwargs)

        tgt_units = split_tgt_meaning_units_sequential(
            src_text,
            tgt_text,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            embed_func=embed_func # Pass the actual embed_func
        )
        if not tgt_units:
            logger.warning("Target units are empty after tokenization.")
            return {'src_units': [], 'tgt_units': [], 'alignments': [], 'status': 'failed', 'error': 'Empty target units'}

        print(f"[DEBUG] tgt_units: {tgt_units}")
        print("[DEBUG] 2. 정렬 시작")
        alignments = align_units(
            src_units, 
            tgt_units,
            embed_func=embed_func
        )
        print(f"[DEBUG] alignments: {alignments}")
        print("[DEBUG] 3. 괄호 처리 시작")
        processed_alignments = process_punctuation(alignments, src_units, tgt_units)
        print(f"[DEBUG] processed_alignments: {processed_alignments}")
        return {
            'src_units': src_units,
            'tgt_units': tgt_units,
            'alignments': processed_alignments,
            'status': 'success'
        }
    except Exception as e:
        import traceback
        print("[ERROR] process_sentence 예외 발생:", e)
        traceback.print_exc()
        logger.error(f"문장 처리 실패: {e}")
        return {
            'src_units': [],
            'tgt_units': [],
            'alignments': [],
            'status': 'failed',
            'error': str(e)
        }

def process_file(
    input_file: str,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    save_results: bool = True,
    output_file: Optional[str] = None,
    openai_model: str = "text-embedding-3-large",
    openai_api_key: Optional[str] = None,
    progress_callback=None,    # 추가
    stop_flag=None,            # 추가
    **kwargs
) -> Optional[pd.DataFrame]:
    """파일 처리 함수 - 진행률 표시 포함"""
    
    logger.info(f"파일 처리 시작: {input_file}")
    
    try:
        # 파일 로드
        io_manager = IOManager()
        df = io_manager.read_file(input_file)
        if df is None or df.empty:
            logger.error(f"파일 로드 실패 또는 파일이 비어 있습니다: {input_file}")
            return None
        
        total_sentences = len(df)
        logger.info(f"처리할 문장 수: {total_sentences}")
        
        results = []
        
        # 메인 진행률 바 추가
        use_callback = progress_callback is not None
        if not use_callback:
            progress_bar = tqdm(
                df.iterrows(), 
                total=total_sentences,
                desc="문장 처리",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100
            )
        
        start_time = time.time()  # 시작 시간 기록
        
        for idx, row in (progress_bar if not use_callback else enumerate(df.iterrows())):
            # 중지 플래그 체크
            if stop_flag and stop_flag.is_set():
                logger.info("사용자 중지 요청, 처리 중단")
                break

            # 진행률 바 설명 업데이트
            if not use_callback:
                progress_bar.set_description(f"문장 {idx+1}/{total_sentences}")
            else:
                progress_callback(idx+1, total_sentences)
            
            try:
                src_text = row.get('src', '') if not use_callback else row[1].get('src', '')
                tgt_text = row.get('tgt', '') if not use_callback else row[1].get('tgt', '')
                print(f"\n[REALTIME] 문장 {idx+1}/{total_sentences}")
                print(f"[REALTIME] src: {src_text}")
                print(f"[REALTIME] tgt: {tgt_text}")

                print("[DEBUG] split_src_meaning_units 진입")
                src_units = split_src_meaning_units(
                    src_text, 
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                print("[DEBUG] split_src_meaning_units 종료")
                print(f"[REALTIME] src_units: {src_units}")

                print("[DEBUG] split_tgt_meaning_units_sequential 진입")
                tgt_units = split_tgt_meaning_units_sequential(
                    src_text,
                    tgt_text,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    embed_func=compute_embeddings_with_cache if use_semantic else None
                )
                print("[DEBUG] split_tgt_meaning_units_sequential 종료")
                print(f"[REALTIME] tgt_units: {tgt_units}")

                # 41번째 문장 src/tgt 별도 저장
                if idx+1 == 41:
                    with open('debug_41_src.txt', 'w', encoding='utf-8') as f:
                        f.write(src_text)
                    with open('debug_41_tgt.txt', 'w', encoding='utf-8') as f:
                        f.write(tgt_text)
                if not src_text.strip() or not tgt_text.strip():
                    logger.warning(f"문장 {idx+1}: 빈 텍스트 - 건너뜀")
                    continue
                # 1. 원문 토크나이징
                print(f"[DEBUG] split_src_meaning_units 진입 (문장 {idx+1})")
                t0 = time.time()
                src_units = split_src_meaning_units(
                    src_text, 
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                print(f"[DEBUG] split_src_meaning_units 반환 (문장 {idx+1}, {len(src_units)}개, {time.time()-t0:.2f}s): {src_units}")
                # 2. 번역문 토크나이징  
                print(f"[DEBUG] split_tgt_meaning_units_sequential 진입 (문장 {idx+1})")
                t0 = time.time()
                tgt_units = split_tgt_meaning_units_sequential(
                    src_text,
                    tgt_text,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    embed_func=compute_embeddings_with_cache if use_semantic else None
                )
                print(f"[DEBUG] split_tgt_meaning_units_sequential 반환 (문장 {idx+1}, {len(tgt_units)}개, {time.time()-t0:.2f}s): {tgt_units}")
                # 3. 정렬
                print(f"[DEBUG] align_tokens 진입 (문장 {idx+1})")
                t0 = time.time()
                embed_func = get_embedder(
                    embedder_name=kwargs.get("embedder_name", "bge"),
                    openai_model=openai_model,
                    openai_api_key=openai_api_key,
                    device=kwargs.get("device", "cpu")
                )
                alignments = align_units(
                    src_units,
                    tgt_units,
                    embed_func=embed_func
                )
                print(f"[DEBUG] align_tokens 반환 (문장 {idx+1}, {len(alignments) if alignments else 0}개, {time.time()-t0:.2f}s): {alignments}")
                # 4. 괄호 처리
                print(f"[DEBUG] process_punctuation 진입 (문장 {idx+1})")
                t0 = time.time()
                alignments = process_punctuation(alignments, src_units, tgt_units)
                print(f"[DEBUG] process_punctuation 반환 (문장 {idx+1}, {len(alignments) if alignments else 0}개, {time.time()-t0:.2f}s): {alignments}")
                # 결과 저장
                row_result = {
                    'id': row.get('id', idx+1) if not use_callback else row[1].get('id', idx+1),
                    'src': src_text,
                    'tgt': tgt_text,
                    'src_units': src_units,
                    'tgt_units': tgt_units,
                    'alignments': alignments,
                    'src_count': len(src_units),
                    'tgt_count': len(tgt_units),
                    'alignment_count': len(alignments) if alignments else 0,
                    'status': 'success'
                }
                
                results.append(row_result)
                
                # 성공 로그 (조용히)
                if (idx + 1) % 10 == 0:  # 10개마다 로그
                    logger.info(f"문장 {idx+1}/{total_sentences} 처리 완료")
                
                # 진행률 바 상태 업데이트
                if not use_callback:
                    success_count = len(results)
                    progress_bar.set_postfix_str(f"성공: {success_count}")
                
            except Exception as e:
                logger.error(f"문장 {idx+1} 처리 실패: {e}")
                
                # 실패한 경우도 결과에 추가
                row_result = {
                    'id': row.get('id', idx+1) if not use_callback else row[1].get('id', idx+1),
                    'src': row.get('src', '') if not use_callback else row[1].get('src', ''),
                    'tgt': row.get('tgt', '') if not use_callback else row[1].get('tgt', ''),
                    'src_units': [],
                    'tgt_units': [],
                    'alignments': [],
                    'src_count': 0,
                    'tgt_count': 0,
                    'alignment_count': 0,
                    'status': f'failed: {str(e)[:50]}'
                }
                results.append(row_result)
                
                if not use_callback:
                    progress_bar.set_postfix_str(f"실패: {str(e)[:20]}...")
        # 진행률 바 완료
        if not use_callback:
            progress_bar.close()
        
        end_time = time.time()  # 종료 시간 기록
        
        if not results:
            logger.error("처리된 결과가 없습니다")
            return None
        
        # DataFrame 생성
        results_df = pd.DataFrame(results)
        
        # 결과 저장
        if save_results:
            if output_file is None:
                output_file = input_file.replace('.xlsx', '_results.xlsx')
            
            print(f"\n저장 중: {output_file}")
            try:
                io_manager.write_df_to_file(results_df, output_file)
                logger.info(f"결과 저장 완료: {output_file}")
            except Exception as e:
                logger.error(f"결과 저장 실패: {output_file}: {e}")
                raise
        
        # 최종 통계
        success_count = len(results_df[results_df['status'] == 'success'])
        total_processed = len(results_df)
        
        print(f"\n처리 완료 요약:")
        print(f"   전체 문장: {total_sentences}")
        print(f"   성공: {success_count}")
        print(f"   실패: {total_processed - success_count}")
        print(f"   성공률: {success_count/total_processed*100:.1f}%")
        print(f"처리 시간: {end_time - start_time:.2f}초")  # 처리 시간 출력
        
        return results_df
        
    except Exception as e:
        logger.error(f"파일 처리 실패: {e}")
        return None

def process_file_with_modules(
    input_file: str,
    output_file: str,
    tokenizer_module,
    embedder_module,
    embedder_name: str,  # 이 줄 추가!
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    openai_model: str = "text-embedding-3-large",
    openai_api_key: Optional[str] = None,
    **kwargs
):
    """모듈을 동적으로 받아서 처리하는 함수 - 진행률 표시 포함"""
    
    logger.info(f"동적 모듈로 파일 처리: {input_file}")
    
    try:
        # 동적 함수 가져오기
        split_src = tokenizer_module.split_src_meaning_units
        split_tgt = tokenizer_module.split_tgt_meaning_units_sequential # Use sequential for consistency
        
        embed_func = None
        if use_semantic:
            if embedder_name == "openai":
                embed_func = embedder_module.get_embedder(model_name=openai_model, openai_api_key=openai_api_key, **kwargs)
            else:
                embed_func = embedder_module.get_embed_func(device_id=kwargs.get("device", "cpu"))
        
        io_manager = IOManager()
        df = io_manager.read_file(input_file)
        if df is None or df.empty:
            logger.error(f"파일 로드 실패 또는 파일이 비어 있습니다: {input_file}")
            return None
    
        total_sentences = len(df)
        logger.info(f"처리할 문장 수: {total_sentences}")
        
        results = []
        
        # 메인 진행률 바 추가
        progress_bar = tqdm(
            df.iterrows(), 
            total=total_sentences,
            desc="동적 모듈 처리",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        start_time = time.time()  # 시작 시간 기록
        
        for idx, row in progress_bar:
            progress_bar.set_description(f"문장 {idx+1}/{total_sentences}")
            
            try:
                src_text = row.get('source', '')
                tgt_text = row.get('target', '')
                
                if not src_text.strip() or not tgt_text.strip():
                    logger.warning(f"문장 {idx+1}: 빈 텍스트 - 건너뜀")
                    continue
                
                # 동적 토크나이저 사용
                progress_bar.set_postfix_str("토크나이징...")
                try:
                    src_units = split_src(src_text, min_tokens, max_tokens)
                    tgt_units = split_tgt(
                        src_text, tgt_text,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    logger.error(f"Error during tokenization for sentence {idx+1}: {e}")
                    raise

                # 동적 임베더로 정렬
                progress_bar.set_postfix_str("정렬...")
                try:
                    alignments = align_units(
                        src_units,
                        tgt_units,
                        embed_func=embed_func
                    )
                except Exception as e:
                    logger.error(f"Error during alignment for sentence {idx+1}: {e}")
                    raise

                results.append({
                    'id': row.get('id', idx+1),
                    'src': src_text, 'tgt': tgt_text,
                    'src_units': src_units, 'tgt_units': tgt_units,
                    'alignments': alignments,
                    'src_count': len(src_units), 'tgt_count': len(tgt_units),
                    'alignment_count': len(alignments), 'status': 'success'
                })
                
                progress_bar.set_postfix_str(f"성공: {len(results)}")
                
            except Exception as e:
                logger.error(f"문장 {idx+1} 처리 실패: {e}")
                progress_bar.set_postfix_str(f"실패: {str(e)[:20]}...")
        
        progress_bar.close()
        
        # 저장
        io_manager = IOManager()
        results_df = pd.DataFrame(results)
        io_manager.write_df_to_file(results_df, output_file)
        
        end_time = time.time()  # 종료 시간 기록
        
        print(f"\n동적 처리 완료: {len(results)}개 문장")
        print(f"처리 시간: {end_time - start_time:.2f}초")  # 처리 시간 출력
        return results_df
        
    except Exception as e:
        logger.error(f"동적 처리 실패: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # 테스트
    test_file = "test_data.xlsx"
    results = process_file(test_file)
    
    if results is not None:
        print("처리 성공")
        print(results.head())
    else:
        print("처리 실패")