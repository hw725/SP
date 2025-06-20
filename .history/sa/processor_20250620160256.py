"""파일 처리 메인 프로세서"""

import io_utils  # 상대 임포트 → 절대 임포트로 변경
from tokenizer import split_src_meaning_units, split_tgt_meaning_units
from aligner import align_tokens_with_embeddings
from embedder import get_embeddings
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def process_file(
    input_file: str,
    output_file: str,
    use_semantic: bool = True,
    min_tokens: int = 2,
    max_tokens: int = 10
) -> bool:
    """
    파일 전체 처리 파이프라인
    
    Args:
        input_file: 입력 엑셀 파일 경로
        output_file: 출력 엑셀 파일 경로
        use_semantic: 의미 기반 분할 사용 여부
        min_tokens: 최소 토큰 수
        max_tokens: 최대 토큰 수
    
    Returns:
        bool: 처리 성공 여부
    """
    try:
        logger.info(f"📁 파일 처리 시작: {input_file}")
        
        # 1. 파일 로드
        df = io_utils.load_excel_file(input_file)
        if df is None:
            logger.error(f"❌ 파일 로드 실패: {input_file}")
            return False
        
        logger.info(f"📊 처리할 문장 수: {len(df)}")
        
        results = []
        
        # 2. 각 문장 처리
        for idx, row in df.iterrows():
            try:
                src_text = str(row['src']).strip()
                tgt_text = str(row['tgt']).strip()
                
                if not src_text or not tgt_text:
                    logger.warning(f"⚠️ 행 {idx}: 빈 텍스트 발견")
                    continue
                
                logger.info(f"🔤 문장 {idx+1} 처리 중...")
                
                # 원문 분할
                src_units = split_src_meaning_units(
                    src_text,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens
                )
                
                # 번역문 분할
                if use_semantic:
                    # 실제 임베딩 사용
                    tgt_units = split_tgt_meaning_units(
                        src_text, tgt_text,
                        embed_func=get_embeddings,
                        use_semantic=True,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens
                    )
                else:
                    # 단순 분할
                    tgt_units = split_tgt_meaning_units(
                        src_text, tgt_text,
                        embed_func=None,
                        use_semantic=False,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens
                    )
                
                # 정렬 수행
                if use_semantic:
                    alignments = align_tokens_with_embeddings(
                        src_units, tgt_units,
                        src_text, tgt_text
                    )
                else:
                    # 단순 정렬 (순서대로)
                    alignments = []
                    min_len = min(len(src_units), len(tgt_units))
                    for i in range(min_len):
                        alignments.append({
                            'src_idx': i,
                            'tgt_idx': i,
                            'src_text': src_units[i],
                            'tgt_text': tgt_units[i],
                            'confidence': 0.5  # 기본값
                        })
                
                # 결과 저장
                result = {
                    'id': row.get('id', idx),
                    'src': src_text,
                    'tgt': tgt_text,
                    'src_units': str(src_units),
                    'tgt_units': str(tgt_units),
                    'alignments': str(alignments),
                    'src_count': len(src_units),
                    'tgt_count': len(tgt_units),
                    'alignment_count': len(alignments)
                }
                
                results.append(result)
                logger.info(f"✅ 문장 {idx+1} 처리 완료: {len(src_units)}→{len(tgt_units)} ({len(alignments)}정렬)")
                
            except Exception as e:
                logger.error(f"❌ 문장 {idx+1} 처리 실패: {e}")
                # 실패한 경우라도 기본 정보는 저장
                results.append({
                    'id': row.get('id', idx),
                    'src': str(row.get('src', '')),
                    'tgt': str(row.get('tgt', '')),
                    'src_units': '[]',
                    'tgt_units': '[]',
                    'alignments': '[]',
                    'src_count': 0,
                    'tgt_count': 0,
                    'alignment_count': 0,
                    'error': str(e)
                })
        
        # 3. 결과 저장
        if results:
            success = io_utils.save_alignment_results(results, output_file)
            if success:
                logger.info(f"✅ 결과 저장 완료: {output_file}")
                logger.info(f"📊 처리 완료: {len(results)}개 문장")
                return True
            else:
                logger.error(f"❌ 결과 저장 실패: {output_file}")
                return False
        else:
            logger.error("❌ 처리된 결과가 없습니다")
            return False
            
    except Exception as e:
        logger.error(f"❌ 파이프라인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_single_sentence(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    min_tokens: int = 2,
    max_tokens: int = 10
) -> Optional[dict]:
    """
    단일 문장 처리
    
    Args:
        src_text: 원문
        tgt_text: 번역문
        use_semantic: 의미 기반 분할 사용 여부
        min_tokens: 최소 토큰 수
        max_tokens: 최대 토큰 수
    
    Returns:
        dict: 처리 결과 또는 None
    """
    try:
        logger.info(f"🔤 단일 문장 처리: {src_text[:50]}...")
        
        # 원문 분할
        src_units = split_src_meaning_units(
            src_text,
            min_tokens=min_tokens,
            max_tokens=max_tokens
        )
        
        # 번역문 분할
        if use_semantic:
            tgt_units = split_tgt_meaning_units(
                src_text, tgt_text,
                embed_func=get_embeddings,
                use_semantic=True,
                min_tokens=min_tokens,
                max_tokens=max_tokens
            )
            
            # 정렬
            alignments = align_tokens_with_embeddings(
                src_units, tgt_units,
                src_text, tgt_text
            )
        else:
            tgt_units = split_tgt_meaning_units(
                src_text, tgt_text,
                embed_func=None,
                use_semantic=False,
                min_tokens=min_tokens,
                max_tokens=max_tokens
            )
            
            # 단순 정렬
            alignments = []
            min_len = min(len(src_units), len(tgt_units))
            for i in range(min_len):
                alignments.append({
                    'src_idx': i,
                    'tgt_idx': i,
                    'src_text': src_units[i],
                    'tgt_text': tgt_units[i],
                    'confidence': 0.5
                })
        
        result = {
            'src': src_text,
            'tgt': tgt_text,
            'src_units': src_units,
            'tgt_units': tgt_units,
            'alignments': alignments,
            'src_count': len(src_units),
            'tgt_count': len(tgt_units),
            'alignment_count': len(alignments)
        }
        
        logger.info(f"✅ 단일 문장 처리 완료: {len(src_units)}→{len(tgt_units)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 단일 문장 처리 실패: {e}")
        return None

if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    test_src = "興也라"
    test_tgt = "興이다."
    
    result = process_single_sentence(test_src, test_tgt, use_semantic=False)
    if result:
        print("✅ 테스트 성공")
        print(f"원문 분할: {result['src_units']}")
        print(f"번역 분할: {result['tgt_units']}")
        print(f"정렬 결과: {result['alignments']}")
    else:
        print("❌ 테스트 실패")
