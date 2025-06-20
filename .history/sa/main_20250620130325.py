# """메인 실행 파일 - 출력 포맷 옵션 추가"""
import argparse
import logging
import os
from src.config import Config
from src.orchestrator import run_processing
from src.components import list_available_tokenizers, list_available_embedders

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 로깅 레벨 조정
    if not verbose:
        logging.getLogger('src.text_alignment').setLevel(logging.WARNING)
        logging.getLogger('soynlp').setLevel(logging.WARNING)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alignment.log', mode='w', encoding='utf-8')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Prototype02 + 형태소 분석기 텍스트 정렬 시스템")
    parser.add_argument("input_path", help="입력 파일 경로 (.xlsx 또는 .csv)")
    parser.add_argument("output_path", help="출력 파일 경로")
    
    # 토크나이저 옵션
    parser.add_argument(
        "--tokenizer", 
        default="default",
        choices=list_available_tokenizers(),
        help="토크나이저 선택 (기본값: prototype02-mecab)"
    )
    
    parser.add_argument("--source-tokenizer", help="원문 전용 토크나이저")
    parser.add_argument("--target-tokenizer", help="번역문 전용 토크나이저")
    
    # 간편 옵션들
    parser.add_argument("--analyzer", choices=['mecab', 'okt', 'komoran', 'hannanum', 'kkma', 'kiwi', 'jieba'], 
                       help="형태소 분석기 간편 선택")
    
    parser.add_argument(
        "--embedder", 
        default="bge-m3",
        choices=list_available_embedders(),
        help="임베더 선택 (기본값: bge-m3)"
    )
    
    # *** 출력 포맷 옵션 추가 ***
    parser.add_argument(
        "--output-format",
        default="phrase_unit",
        choices=["phrase_unit", "compact"],
        help="출력 포맷 (기본값: phrase_unit - 구 단위 분할, compact - 기존 형태)"
    )
    
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=4, help="병렬 처리 워커 수")
    parser.add_argument("--chunk-size", type=int, default=50, help="청크 크기")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not os.path.exists(args.input_path):
        logging.error(f"입력 파일을 찾을 수 없습니다: {args.input_path}")
        return
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"출력 디렉토리 생성: {output_dir}")
        except Exception as e:
            logging.error(f"출력 디렉토리 생성 실패: {e}")
            return
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 토크나이저 설정 로직
    if args.analyzer:
        source_tokenizer_type = f"prototype02-{args.analyzer}"
        target_tokenizer_type = "prototype02"
    else:
        source_tokenizer_type = args.source_tokenizer or args.tokenizer
        target_tokenizer_type = args.target_tokenizer or "prototype02"
    
    # 설정 생성
    config = Config(
        input_path=args.input_path,
        output_path=args.output_path,
        source_tokenizer_type=source_tokenizer_type,
        target_tokenizer_type=target_tokenizer_type,
        embedder_type=args.embedder,
        output_format=args.output_format,  # 출력 포맷 설정
        use_parallel=args.parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )
    
    # 실행
    logger.info("=== Prototype02 + 형태소 분석기 파이프라인 시작 ===")
    logger.info(f"입력: {args.input_path}")
    logger.info(f"출력: {args.output_path}")
    logger.info(f"출력 포맷: {args.output_format}")
    logger.info(f"토크나이저: {source_tokenizer_type} (원문), {target_tokenizer_type} (번역문)")
    logger.info(f"임베더: {args.embedder}")
    logger.info(f"병렬 처리: {args.parallel}")
    
    try:
        run_processing(config)
        logger.info("=== 처리 완료 ===")
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        raise

if __name__ == "__main__":
    main()