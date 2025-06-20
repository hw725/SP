"""문장 단위 처리 및 정렬 모듈"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from io_utils import load_excel_file as load_excel, save_alignment_results as save_excel
from sa_tokenizers import split_src_meaning_units, split_tgt_meaning_units  # 🔧 수정
from sa_embedders import compute_embeddings_with_cache  # 🔧 수정
from aligner import align_tokens_with_embeddings as align_tokens
from punctuation import process_punctuation

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
    file_path: str,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    save_results: bool = True,
    output_file: Optional[str] = None,
    **kwargs
) -> Optional[pd.DataFrame]:
    """파일 단위 처리"""
    
    logger.info(f"📁 파일 처리 시작: {file_path}")
    
    try:
        # 데이터 로드
        df = load_excel(file_path)
        logger.info(f"📊 처리할 문장 수: {len(df)}")
        
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"🔤 문장 {idx+1} 처리 중...")
            
            src_text = row.get('src', '')
            tgt_text = row.get('tgt', '')
            
            if not src_text or not tgt_text:
                logger.warning(f"⚠️ 문장 {idx+1}: 빈 텍스트")
                continue
            
            # 문장 처리
            result = process_sentence(
                src_text,
                tgt_text,
                use_semantic=use_semantic,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # 결과 저장
            row_result = {
                'id': row.get('id', idx+1),
                'src': src_text,
                'tgt': tgt_text,
                'src_units': result['src_units'],
                'tgt_units': result['tgt_units'],
                'alignments': result['alignments'],
                'src_count': len(result['src_units']),
                'tgt_count': len(result['tgt_units']),
                'alignment_count': len(result['alignments']),
                'status': result['status']
            }
            
            if result['status'] == 'failed':
                row_result['error'] = result.get('error', '')
            
            results.append(row_result)
            
            logger.info(f"✅ 문장 {idx+1} 처리 완료: {len(result['src_units'])}→{len(result['tgt_units'])} ({len(result['alignments'])}정렬)")
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame(results)
        
        # 결과 저장
        if save_results:
            if output_file is None:
                output_file = file_path.replace('.xlsx', '_results.xlsx')
            
            save_excel(results_df, output_file)
            logger.info(f"✅ 결과 저장 완료: {output_file}")
        
        logger.info(f"📊 처리 완료: {len(results)}개 문장")
        
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
    parallel: bool = False,
    **kwargs
):
    """모듈을 동적으로 받아서 처리"""
    
    logger.info(f"📁 파일 처리 시작: {input_file}")
    
    try:
        # 동적 함수 가져오기
        split_src_meaning_units = tokenizer_module.split_src_meaning_units
        split_tgt_meaning_units = tokenizer_module.split_tgt_meaning_units
        compute_embeddings_with_cache = embedder_module.compute_embeddings_with_cache
        
        # 기존 process_file 로직과 동일하지만 동적 함수 사용
        from io_utils import load_excel_file, save_alignment_results
        
        df = load_excel_file(input_file)
        if df is None:
            return None
            
        logger.info(f"📊 처리할 문장 수: {len(df)}")
        
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"🔤 문장 {idx+1} 처리 중...")
            
            src_text = row.get('src', '')
            tgt_text = row.get('tgt', '')
            
            if not src_text or not tgt_text:
                logger.warning(f"⚠️ 문장 {idx+1}: 빈 텍스트")
                continue
            
            # 토크나이징 (동적 함수 사용)
            src_units = split_src_meaning_units(
                src_text, 
                min_tokens=min_tokens, 
                max_tokens=max_tokens
            )
            
            tgt_units = split_tgt_meaning_units(
                src_text,
                tgt_text,
                use_semantic=use_semantic,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                embed_func=compute_embeddings_with_cache if use_semantic else None
            )
            
            # 정렬 (동적 임베더 사용)
            from aligner import align_tokens
            alignments = align_tokens(
                src_units, 
                tgt_units,
                embed_func=compute_embeddings_with_cache
            )
            
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
                'alignment_count': len(alignments),
                'status': 'success'
            }
            
            results.append(row_result)
            
            logger.info(f"✅ 문장 {idx+1} 처리 완료: {len(src_units)}→{len(tgt_units)} ({len(alignments)}정렬)")
        
        # 결과 저장
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        if save_alignment_results(results_df, output_file):
            logger.info(f"✅ 결과 저장 완료: {output_file}")
            return results_df
        else:
            return None
        
    except Exception as e:
        logger.error(f"❌ 파일 처리 실패: {e}")
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
