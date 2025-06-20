"""문장 단위 처리 및 정렬 모듈 - 진행률 표시 포함"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm  # 진행률 표시 추가
from io_utils import load_excel_file as load_excel, save_alignment_results as save_excel
from sa_tokenizers import split_src_meaning_units, split_tgt_meaning_units
from sa_embedders import compute_embeddings_with_cache
from aligner import align_tokens_with_embeddings as align_tokens

# punctuation import 안전 처리
try:
    from punctuation import process_punctuation
except ImportError:
    def process_punctuation(alignments, src_units, tgt_units):
        return alignments

logger = logging.getLogger(__name__)

def process_sentence(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """단일 문장 처리"""
    
    try:
        # 1. 토크나이징
        src_units = split_src_meaning_units(
            src_text, 
            min_tokens=min_tokens, 
            max_tokens=max_tokens,
            use_advanced=True
        )
        
        tgt_units = split_tgt_meaning_units(
            src_text,
            tgt_text,
            use_semantic=use_semantic,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            embed_func=compute_embeddings_with_cache if use_semantic else None
        )
        
        # 2. 정렬
        alignments = align_tokens(
            src_units, 
            tgt_units,
            embed_func=compute_embeddings_with_cache
        )
        
        # 3. 괄호 처리
        processed_alignments = process_punctuation(alignments, src_units, tgt_units)
        
        return {
            'src_units': src_units,
            'tgt_units': tgt_units,
            'alignments': processed_alignments,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"❌ 문장 처리 실패: {e}")
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
    **kwargs
) -> Optional[pd.DataFrame]:
    """파일 처리 함수 - 진행률 표시 포함"""
    
    logger.info(f"📁 파일 처리 시작: {input_file}")
    
    try:
        # 파일 로드
        df = load_excel(input_file)
        if df is None:
            logger.error(f"❌ 파일 로드 실패: {input_file}")
            return None
        
        total_sentences = len(df)
        logger.info(f"📊 처리할 문장 수: {total_sentences}")
        
        results = []
        
        # 🎯 메인 진행률 바 추가
        progress_bar = tqdm(
            df.iterrows(), 
            total=total_sentences,
            desc="🔤 문장 처리",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for idx, row in progress_bar:
            # 진행률 바 설명 업데이트
            progress_bar.set_description(f"🔤 문장 {idx+1}/{total_sentences}")
            
            try:
                src_text = row.get('src', '')
                tgt_text = row.get('tgt', '')
                
                if not src_text or not tgt_text:
                    logger.warning(f"⚠️ 문장 {idx+1}: 빈 텍스트 - 건너뜀")
                    continue
                
                # 1. 원문 토크나이징
                progress_bar.set_postfix_str("원문 토크나이징...")
                src_units = split_src_meaning_units(
                    src_text, 
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
                
                # 2. 번역문 토크나이징  
                progress_bar.set_postfix_str("번역문 토크나이징...")
                tgt_units = split_tgt_meaning_units(
                    src_text,
                    tgt_text,
                    use_semantic=use_semantic,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    embed_func=compute_embeddings_with_cache if use_semantic else None
                )
                
                # 3. 정렬
                progress_bar.set_postfix_str("토큰 정렬...")
                alignments = align_tokens(
                    src_units, 
                    tgt_units,
                    embed_func=compute_embeddings_with_cache
                )
                
                # 4. 괄호 처리
                progress_bar.set_postfix_str("괄호 처리...")
                alignments = process_punctuation(alignments, src_units, tgt_units)
                
                # 결과 저장
                row_result = {
                    'id': row.get('id', idx+1),
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
                    logger.info(f"✅ {idx+1}/{total_sentences} 문장 처리 완료")
                
                # 진행률 바 상태 업데이트
                success_count = len(results)
                progress_bar.set_postfix_str(f"성공: {success_count}")
                
            except Exception as e:
                logger.error(f"❌ 문장 {idx+1} 처리 실패: {e}")
                
                # 실패한 경우도 결과에 추가
                row_result = {
                    'id': row.get('id', idx+1),
                    'src': row.get('src', ''),
                    'tgt': row.get('tgt', ''),
                    'src_units': [],
                    'tgt_units': [],
                    'alignments': [],
                    'src_count': 0,
                    'tgt_count': 0,
                    'alignment_count': 0,
                    'status': f'failed: {str(e)[:50]}'
                }
                results.append(row_result)
                
                progress_bar.set_postfix_str(f"실패: {str(e)[:20]}...")
        
        # 진행률 바 완료
        progress_bar.close()
        
        if not results:
            logger.error("❌ 처리된 결과가 없습니다")
            return None
        
        # DataFrame 생성
        results_df = pd.DataFrame(results)
        
        # 결과 저장
        if save_results:
            if output_file is None:
                output_file = input_file.replace('.xlsx', '_results.xlsx')
            
            print(f"\n💾 결과 저장 중: {output_file}")
            if save_excel(results_df, output_file):
                logger.info(f"✅ 결과 저장 완료: {output_file}")
            else:
                logger.error(f"❌ 결과 저장 실패: {output_file}")
        
        # 최종 통계
        success_count = len(results_df[results_df['status'] == 'success'])
        total_processed = len(results_df)
        
        print(f"\n🎉 처리 완료 요약:")
        print(f"   📊 전체 문장: {total_sentences}")
        print(f"   ✅ 성공: {success_count}")
        print(f"   ❌ 실패: {total_processed - success_count}")
        print(f"   📈 성공률: {success_count/total_processed*100:.1f}%")
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ 파일 처리 실패: {e}")
        return None

def process_file_with_modules(
    input_file: str,
    output_file: str,
    tokenizer_module,
    embedder_module,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    **kwargs
):
    """모듈을 동적으로 받아서 처리하는 함수 - 진행률 표시 포함"""
    
    logger.info(f"📁 동적 모듈로 파일 처리: {input_file}")
    
    try:
        # 동적 함수 가져오기
        split_src = tokenizer_module.split_src_meaning_units
        split_tgt = tokenizer_module.split_tgt_meaning_units
        embed_func = embedder_module.compute_embeddings_with_cache
        
        from io_utils import load_excel_file, save_alignment_results
        
        df = load_excel_file(input_file)
        if df is None:
            return None
        
        total_sentences = len(df)
        logger.info(f"📊 처리할 문장 수: {total_sentences}")
        
        results = []
        
        # 🎯 메인 진행률 바 추가
        progress_bar = tqdm(
            df.iterrows(), 
            total=total_sentences,
            desc="🔤 동적 모듈 처리",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for idx, row in progress_bar:
            progress_bar.set_description(f"🔤 문장 {idx+1}/{total_sentences}")
            
            try:
                src_text = row.get('src', '')
                tgt_text = row.get('tgt', '')
                
                if not src_text or not tgt_text:
                    continue
                
                # 동적 토크나이저 사용
                progress_bar.set_postfix_str("토크나이징...")
                src_units = split_src(src_text, min_tokens, max_tokens)
                tgt_units = split_tgt(
                    src_text, tgt_text, 
                    use_semantic=use_semantic,
                    embed_func=embed_func if use_semantic else None
                )
                
                # 동적 임베더로 정렬
                progress_bar.set_postfix_str("정렬...")
                from aligner import align_tokens
                alignments = align_tokens(src_units, tgt_units, embed_func=embed_func)
                
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
                logger.error(f"❌ 문장 {idx+1} 처리 실패: {e}")
                progress_bar.set_postfix_str(f"실패: {str(e)[:20]}...")
        
        progress_bar.close()
        
        # 저장
        import pandas as pd
        results_df = pd.DataFrame(results)
        save_alignment_results(results_df, output_file)
        
        print(f"\n🎉 동적 처리 완료: {len(results)}개 문장")
        return results_df
        
    except Exception as e:
        logger.error(f"❌ 동적 처리 실패: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # 테스트
    test_file = "test_data.xlsx"
    results = process_file(test_file)
    
    if results is not None:
        print("✅ 처리 성공")
        print(results.head())
    else:
        print("❌ 처리 실패")
