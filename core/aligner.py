"""
PA (Paragraph Aligner) - 통합된 아키텍처로 순차적/의미적 정렬 지원
- 설정 기반 동작 방식 선택 (sequential, semantic, hybrid)
- 강화된 에러 처리 및 무결성 보장
- 실제 품질 메트릭 및 사용자 경험 개선
"""
import sys
import os
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# 필수 패키지
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from difflib import SequenceMatcher
except ImportError as e:
    raise ImportError(f"필수 패키지 누락: {e}. pip install pandas numpy tqdm 실행 필요")

# 선택적 패키지
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# 프로젝트 내부 모듈
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from .io_utils import IOManager
    from .sentence_splitter import split_target_sentences_advanced, split_source_by_meaning_units
except ImportError as e:
    raise ImportError(f"필수 core 모듈 로드 실패: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pa_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# SA 임베더 - 선택적 import
try:
    from .embedders import get_embedder as get_embedding_function
    from .processor import align_units
    SA_EMBEDDER_AVAILABLE = True
except ImportError:
    SA_EMBEDDER_AVAILABLE = False
    logger.warning("SA 임베더를 사용할 수 없습니다. 순차적 정렬만 지원됩니다.")


class AlignmentMode(Enum):
    """정렬 모드 열거형"""
    SEQUENTIAL = "sequential"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class AlignmentConfig:
    """정렬 설정 클래스"""
    mode: str = "hybrid"  # sequential, semantic, hybrid
    embedder_name: str = "bge"  # bge, openai
    device: str = "cpu"  # cpu, cuda
    openai_model: str = "text-embedding-3-large"
    openai_api_key: Optional[str] = None
    
    # 품질 임계값
    similarity_threshold: float = 0.3
    hybrid_threshold: float = 0.8
    
    # 가중치
    sequential_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # 기타 설정
    verify_integrity: bool = True
    log_quality_stats: bool = True
    progress_bar: bool = True


@dataclass
class AlignmentResult:
    """정렬 결과 클래스"""
    aligned_pairs: List[Dict]
    quality_score: float
    method_used: str
    processing_time: float
    integrity_verified: bool = False
    stats: Optional[Dict] = None


class IntegrityError(Exception):
    """무결성 검증 실패 예외"""
    pass


class ParagraphAligner:
    """통합 단락 정렬기"""
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.embedder = None
        self._initialize_embedder()
    
    def _initialize_embedder(self) -> None:
        """임베더 초기화"""
        if self.config.mode in ["semantic", "hybrid"] and SA_EMBEDDER_AVAILABLE:
            try:
                if self.config.embedder_name == "openai":
                    self.embedder = get_embedding_function(
                        self.config.embedder_name,
                        device_id=self.config.device,
                        model_name=self.config.openai_model,
                        openai_api_key=self.config.openai_api_key
                    )
                else:
                    self.embedder = get_embedding_function(
                        self.config.embedder_name,
                        device_id=self.config.device
                    )
            except (ValueError, ImportError) as e:
                logger.error(f"임베더 초기화 실패: {e}")
                if self.config.mode == "semantic":
                    self.config.mode = "sequential"
                    logger.warning("의미적 정렬에서 순차적 정렬로 fallback")
    
    
    
    def align_sequential(self, source_text: str, target_sentences: List[str]) -> Tuple[List[Dict], float]:
        """순차적 정렬 (구두점 기반)"""
        if not target_sentences:
            return [], 0.0

        try:
            aligned_chunks = split_source_by_meaning_units(source_text, len(target_sentences))
            
            result = []
            for i, (src, tgt) in enumerate(zip(aligned_chunks, target_sentences)):
                result.append({
                    'id': i + 1,
                    'source': src.strip(),
                    'target': tgt.strip(),
                    'method': 'sequential'
                })
            
            # 간단한 품질 점수 (길이 기반)
            avg_src_len = np.mean([len(pair['source']) for pair in result])
            avg_tgt_len = np.mean([len(pair['target']) for pair in result])
            quality_score = min(avg_src_len, avg_tgt_len) / (max(avg_src_len, avg_tgt_len) if max(avg_src_len, avg_tgt_len) > 0 else 1.0)
            
            return result, quality_score
            
        except Exception as e:
            logger.error(f"순차적 정렬 실패: {e}")
            return [], 0.0
    
    def align_semantic(self, source_text: str, target_sentences: List[str]) -> Tuple[List[Dict], float]:
        """의미적 정렬 (임베딩 기반)"""
        if not self.embedder:
            logger.warning("임베더가 없어 순차적 정렬로 fallback")
            return self.align_sequential(source_text, target_sentences)
        
        try:
            # 원문을 의미 단위로 분할
            src_sentences = split_target_sentences_advanced(source_text)
            
            if not src_sentences or not target_sentences:
                return self.align_sequential(source_text, target_sentences)
            
            # 정렬 수행
            aligned_pairs = align_units(
                src_sentences,
                target_sentences,
                embed_func=self.embedder,
                similarity_threshold=self.config.similarity_threshold
            )

            # 결과 형식 맞추기
            result = []
            for pair in aligned_pairs:
                result.append({
                    'id': pair['tgt_idx'] + 1,
                    'source': pair['src'],
                    'target': pair['tgt'],
                    'similarity': pair['score'],
                    'method': 'semantic'
                })

            # 품질 점수 계산
            valid_similarities = [pair.get('similarity', 0) for pair in result if pair.get('similarity', 0) > 0]
            quality_score = np.mean(valid_similarities) if valid_similarities else 0.0
            
            return result, quality_score
            
        except Exception as e:
            logger.error(f"의미적 정렬 실패: {e}")
            return self.align_sequential(source_text, target_sentences)
    
    def _merge_alignment_results(
        self, 
        sequential_result: Tuple[List[Dict], float],
        semantic_result: Tuple[List[Dict], float]
    ) -> Tuple[List[Dict], float, str]:
        """하이브리드 정렬: 두 결과를 가중 평균으로 결합"""
        seq_pairs, seq_score = sequential_result
        sem_pairs, sem_score = semantic_result
        
        # 가중 품질 점수
        combined_score = (
            self.config.sequential_weight * seq_score + 
            self.config.semantic_weight * sem_score
        )
        
        # 임계값 기반 선택
        if combined_score >= self.config.hybrid_threshold:
            # 높은 품질이면 의미적 정렬 우선
            if sem_score > seq_score:
                return sem_pairs, combined_score, "hybrid_semantic"
            else:
                return seq_pairs, combined_score, "hybrid_sequential"
        else:
            # 낮은 품질이면 더 나은 쪽 선택
            if sem_score > seq_score:
                return sem_pairs, sem_score, "semantic_fallback"
            else:
                return seq_pairs, seq_score, "sequential_fallback"
    
    def _verify_and_repair_integrity(self, source_text: str, target_sentences: List[str], aligned_pairs: List[Dict]) -> Tuple[List[Dict], bool]:
        """Verifies and repairs the integrity of the aligned text by content comparison."""
        if not self.config.verify_integrity:
            return aligned_pairs, True

        try:
            # Reconstruct the source and target text from the aligned pairs
            aligned_src_text = "".join([pair.get('source', '') for pair in aligned_pairs])
            aligned_tgt_text = "".join([pair.get('target', '') for pair in aligned_pairs])

            # Normalize whitespace for a more robust comparison
            original_src_norm = "".join(source_text.split())
            aligned_src_norm = "".join(aligned_src_text.split())
            
            original_tgt_norm = "".join("".join(target_sentences).split())
            aligned_tgt_norm = "".join(aligned_tgt_text.split())

            if original_src_norm != aligned_src_norm or original_tgt_norm != aligned_tgt_norm:
                logger.warning(f"Integrity check failed.")
                logger.debug(f"Original Source (norm): {original_src_norm[:100]}...")
                logger.debug(f"Aligned Source (norm):  {aligned_src_norm[:100]}...")
                logger.debug(f"Original Target (norm): {original_tgt_norm[:100]}...")
                logger.debug(f"Aligned Target (norm):  {aligned_tgt_norm[:100]}...")
                # Simple repair: fall back to sequential alignment
                repaired_pairs, _ = self.align_sequential(source_text, target_sentences)
                return repaired_pairs, False # Indicate that repair was attempted

            return aligned_pairs, True

        except Exception as e:
            logger.error(f"Error during integrity verification: {e}")
            return aligned_pairs, False
    
    def _repair_text_integrity(self, source_text: str, target_sentences: List[str], aligned_pairs: List[Dict]) -> Tuple[List[Dict], bool]:
        """텍스트 무결성 복구"""
        try:
            # 간단한 복구: 순차적 정렬로 재시도
            logger.info("무결성 복구 시도: 순차적 정렬로 재처리")
            repaired_pairs, _ = self.align_sequential(source_text, target_sentences)
            
            # 기존 정렬의 메타데이터 보존
            for i, pair in enumerate(repaired_pairs):
                if i < len(aligned_pairs):
                    pair.update({k: v for k, v in aligned_pairs[i].items() if k not in ['source', 'target']})
                    pair['method'] = f"{pair.get('method', 'sequential')}_repaired"
            
            return repaired_pairs, True
            
        except Exception as e:
            logger.error(f"무결성 복구 실패: {e}")
            return aligned_pairs, False
    
    def _log_quality_statistics(self, result: AlignmentResult) -> None:
        """품질 통계 로깅"""
        if not self.config.log_quality_stats:
            return
        
        try:
            stats = {
                'total_pairs': len(result.aligned_pairs),
                'quality_score': result.quality_score,
                'method_used': result.method_used,
                'processing_time': result.processing_time,
                'integrity_verified': result.integrity_verified
            }
            
            # 추가 통계
            if result.aligned_pairs:
                avg_src_len = np.mean([len(pair.get('source', '')) for pair in result.aligned_pairs])
                avg_tgt_len = np.mean([len(pair.get('target', '')) for pair in result.aligned_pairs])
                empty_sources = sum(1 for pair in result.aligned_pairs if not pair.get('source', '').strip())
                
                stats.update({
                    'avg_source_length': avg_src_len,
                    'avg_target_length': avg_tgt_len,
                    'empty_sources': empty_sources,
                    'empty_source_ratio': empty_sources / len(result.aligned_pairs)
                })
            
            result.stats = stats
            logger.info(f"품질 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"품질 통계 로깅 실패: {e}")
    
    def align_paragraphs(self, source_text: str, target_sentences: List[str]) -> AlignmentResult:
        """메인 정렬 함수"""
        import time
        start_time = time.time()
        
        if not source_text.strip() or not target_sentences:
            return AlignmentResult(
                aligned_pairs=[],
                quality_score=0.0,
                method_used="empty_input",
                processing_time=0.0,
                integrity_verified=True
            )
        
        try:
            mode = AlignmentMode(self.config.mode)
            
            if mode == AlignmentMode.SEQUENTIAL:
                aligned_pairs, quality_score = self.align_sequential(source_text, target_sentences)
                method_used = "sequential"
                
            elif mode == AlignmentMode.SEMANTIC:
                aligned_pairs, quality_score = self.align_semantic(source_text, target_sentences)
                method_used = "semantic"
                
            elif mode == AlignmentMode.HYBRID:
                seq_result = self.align_sequential(source_text, target_sentences)
                sem_result = self.align_semantic(source_text, target_sentences)
                aligned_pairs, quality_score, method_used = self._merge_alignment_results(seq_result, sem_result)
            
            else:
                raise ValueError(f"지원하지 않는 모드: {self.config.mode}")
            
            # 무결성 검증 및 복구
            aligned_pairs, integrity_verified = self._verify_and_repair_integrity(
                source_text, target_sentences, aligned_pairs
            )
            
            processing_time = time.time() - start_time
            
            result = AlignmentResult(
                aligned_pairs=aligned_pairs,
                quality_score=quality_score,
                method_used=method_used,
                processing_time=processing_time,
                integrity_verified=integrity_verified
            )
            
            # 품질 통계 로깅
            self._log_quality_statistics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"정렬 처리 실패: {e}")
            processing_time = time.time() - start_time
            return AlignmentResult(
                aligned_pairs=[],
                quality_score=0.0,
                method_used="error",
                processing_time=processing_time,
                integrity_verified=False
            )


def process_paragraph_file(
    input_file: str,
    output_file: str,
    config: Optional[AlignmentConfig] = None
) -> bool:
    """파일 기반 단락 정렬 처리"""
    if config is None:
        config = AlignmentConfig()
    
    try:
        # 입력 파일 로드
        logger.info(f"입력 파일 로드: {input_file}")
        df = pd.read_excel(input_file)
        
        # 컬럼명 자동 인식 (원문/번역문 또는 source/target)
        col_map = None
        if 'source' in df.columns and 'target' in df.columns:
            col_map = {'source': 'source', 'target': 'target'}
        elif '원문' in df.columns and '번역문' in df.columns:
            col_map = {'source': '원문', 'target': '번역문'}
        else:
            raise ValueError("입력 파일에 'source', 'target' 또는 '원문', '번역문' 컬럼이 필요합니다.")
        
        aligner = ParagraphAligner(config)
        results = []
        
        # 진행상황 표시
        iterator = tqdm(df.iterrows(), total=len(df), desc="단락 정렬") if config.progress_bar else df.iterrows()
        
        for idx, row in iterator:
            source_text = str(row[col_map['source']]) if pd.notna(row[col_map['source']]) else ""
            target_text = str(row[col_map['target']]) if pd.notna(row[col_map['target']]) else ""
            
            if not source_text.strip() or not target_text.strip():
                logger.warning(f"Skipping paragraph {idx+1}: empty source or target text.")
                continue
            
            # 번역문을 문장 단위로 분할
            target_sentences = split_target_sentences_advanced(target_text)
            # [발화동사+인용구] 병합 후처리
            target_sentences = merge_speaker_quote(target_sentences)
            # 정렬 수행
            result = aligner.align_paragraphs(source_text, target_sentences)
            
            # 결과 추가
            for pair in result.aligned_pairs:
                pair.update({
                    'paragraph_id': idx + 1,
                    'quality_score': result.quality_score,
                    'method': result.method_used
                })
                results.append(pair)
        
        # 결과 저장
        if results:
            result_df = pd.DataFrame(results)
            io_manager = IOManager()
            io_manager.write_df_to_file(result_df, output_file)
            logger.info(f"결과 저장 완료: {output_file} ({len(results)}개 정렬 쌍)")
            return True
        else:
            logger.warning("정렬 결과가 없습니다.")
            return False
            
    except Exception as e:
        logger.error(f"파일 처리 실패: {e}")
        return False


def legacy_align_paragraphs(
    target_sentences: List[str],
    source_text: str,
    embed_func=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """레거시 호환성을 위한 래퍼 함수"""
    logger.warning("레거시 함수 사용 중입니다. 새로운 API로 이전을 권장합니다.")
    
    config = AlignmentConfig(
        mode="semantic" if embed_func else "sequential",
        similarity_threshold=similarity_threshold
    )
    
    aligner = ParagraphAligner(config)
    result = aligner.align_paragraphs(source_text, target_sentences)
    
    return result.aligned_pairs


def merge_speaker_quote(sentences: list) -> list:
    """
    [발화동사+인용구] 패턴 병합: 예) '그는 말했다. "나는 ..."' → '그는 말했다. "나는 ..."'
    - 발화동사로 끝나고 다음 문장이 따옴표로 시작하면 병합
    - 한글: 말했다, 하였다, 외쳤다, 물었다, 답했다 등
    - 중국어: 说道, 说, 问道, 回答道 등 (확장 가능)
    """
    import re
    if not sentences or len(sentences) < 2:
        return sentences
    result = []
    i = 0
    # 발화동사 패턴 (한글/중국어/영어 일부)
    verb_pattern = r'(말했다|하였다|외쳤다|물었다|답했다|전했다|고했다|라고 했다|라고 하였다|说道|说|问道|回答道|said|asked|replied|answered)[.。!]?$'
    quote_pattern = r'^["“‘『「]'  # 따옴표류로 시작
    while i < len(sentences):
        cur = sentences[i]
        if i < len(sentences) - 1:
            if re.search(verb_pattern, cur.strip()) and re.match(quote_pattern, sentences[i+1].strip()):
                # 병합
                merged = cur.rstrip() + ' ' + sentences[i+1].lstrip()
                result.append(merged)
                i += 2
                continue
        result.append(cur)
        i += 1
    return result

# 메인 실행 부분
if __name__ == "__main__":
    # 예제 실행
    config = AlignmentConfig(
        mode="hybrid",
        embedder_name="bge",
        device="cpu",
        progress_bar=True,
        log_quality_stats=True
    )
    
    # 테스트 데이터
    source = "이것은 테스트 문장입니다. 여러 문장으로 구성되어 있습니다."
    target = ["This is a test sentence.", "It consists of multiple sentences."]
    
    aligner = ParagraphAligner(config)
    result = aligner.align_paragraphs(source, target)
    
    print(f"정렬 결과: {len(result.aligned_pairs)}개 쌍")
    print(f"품질 점수: {result.quality_score:.3f}")
    print(f"사용된 방법: {result.method_used}")
    
    for pair in result.aligned_pairs:
        print(f"  원문: {pair['source']}")
        print(f"  번역: {pair['target']}")
        print()
