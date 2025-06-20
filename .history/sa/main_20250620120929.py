"""메인 실행 파일 - 기본값 자동 적용"""
import argparse
import logging
from src.config import Config
from src.orchestrator import run_processing
from src.components import list_available_tokenizers, list_available_embedders

def main():
    parser = argparse.ArgumentParser(description="Prototype02 + 분석기 시스템 (기본값: 원문 jieba, 번역문 mecab)")
    parser.add_argument("input_path", help="입력 파일 경로 (.xlsx 또는 .csv)")
    parser.add_argument("output_path", help="출력 파일 경로")
    
    # 토크나이저 옵션 (기본값 자동 적용)
    parser.add_argument(
        "--tokenizer", 
        default="default",  # 원문 jieba, 번역문 mecab
        choices=list_available_tokenizers(),
        help="토크나이저 조합 (기본값: 원문 jieba, 번역문 mecab)"
    )
    
    # 조합형 옵션
    parser.add_argument("--use-combo", action="store_true", 
                       help="번역문에 다중 분석기 사용 (jieba+mecab)")
    
    # 고급 사용자용 개별 설정
    parser.add_argument("--source-tokenizer", help="원문 토크나이저 개별 설정")
    parser.add_argument("--target-tokenizer", help="번역문 토크나이저 개별 설정")
    
    parser.add_argument(
        "--embedder", 
        default="bge-m3",
        choices=list_available_embedders(),
        help="임베더 선택 (기본값: bge-m3)"
    )
    
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=4, help="병렬 처리 워커 수")
    parser.add_argument("--chunk-size", type=int, default=50, help="청크 크기")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # *** 기본값 자동 적용 로직 ***
    if args.use_combo:
        # 조합형 사용
        tokenizer_type = 'default-combo'
        logger.info("다중 분석기 조합 모드 활성화")
    elif args.source_tokenizer or args.target_tokenizer:
        # 개별 설정 사용
        tokenizer_type = args.tokenizer
        logger.info("개별 토크나이저 설정 사용")
    else:
        # 완전 기본값 (아무 옵션 없음)
        tokenizer_type = 'default'  # 원문 jieba, 번역문 mecab
        logger.info("기본값 자동 적용: 원문 jieba, 번역문 mecab")
    
    # 설정 생성 (기본값이 이미 설정됨)
    config = Config(
        input_path=args.input_path,
        output_path=args.output_path,
        source_tokenizer_type=args.source_tokenizer or tokenizer_type,
        target_tokenizer_type=args.target_tokenizer or tokenizer_type,
        embedder_type=args.embedder,
        use_parallel=args.parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )
    
    # 실행
    logger.info("=== Prototype02 + 분석기 자동 시스템 시작 ===")
    logger.info(f"입력: {args.input_path}")
    logger.info(f"출력: {args.output_path}")
    logger.info(f"🎯 기본 설정: 원문 Jieba 분석 + 번역문 MeCab 분석")
    logger.info(f"임베더: {args.embedder}")
    logger.info(f"병렬 처리: {args.parallel}")
    
    try:
        run_processing(config)
        logger.info("=== 처리 완료 ===")
        logger.info("💡 팁: 다른 분석기를 사용하려면 --tokenizer 옵션을 사용하세요")
        logger.info("     예: --tokenizer mecab-jieba 또는 --use-combo")
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        raise

if __name__ == "__main__":
    main()