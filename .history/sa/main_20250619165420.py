# main.py
import argparse
import logging
import os
# import multiprocessing # freeze_support 사용 시 필요
from src.config import Config
from src.orchestrator import run_processing
# from src.components import ALIGNER_REGISTRY, EMBEDDER_REGISTRY, TOKENIZER_REGISTRY # 직접 사용 안 함

# logger 정의 (루트 로거 사용 또는 특정 이름 지정)
# logging.getLogger()는 루트 로거를 가져오며, basicConfig에 의해 설정됨.
# 특정 모듈 로거를 사용하려면 logger = logging.getLogger(__name__) 사용.
# 여기서는 basicConfig가 루트 로거를 설정하므로, 이후 logging.info 등으로 사용 가능.

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    # 기존 핸들러 제거 (특히 Jupyter Notebook 등에서 중복 로깅 방지)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alignment.log', mode='w', encoding='utf-8') # 매번 새로 쓰기
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="텍스트 정렬 파이프라인")
    parser.add_argument("input_path", help="입력 Excel 또는 CSV 파일 경로")
    parser.add_argument("output_path", help="출력 Excel 또는 CSV 파일 경로")
    parser.add_argument("--source_tokenizer", default="jieba", help="원문 토크나이저 타입")
    parser.add_argument("--target_tokenizer", default="mecab", help="번역문 토크나이저 타입")
    parser.add_argument("--embedder", default="sentence-transformer", help="임베더 타입")  # 원하는 기본값으로 변경
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용 여부")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="병렬 처리 시 사용할 워커 수")
    parser.add_argument("--chunk_size", type=int, default=50, help="병렬 처리 시 청크 크기")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 출력 여부")
    args = parser.parse_args()

    setup_logging(args.verbose) # 로깅 설정 먼저 호출
    
    if not os.path.exists(args.input_path):
        logging.error(f"입력 파일을 찾을 수 없습니다: {args.input_path}")
        return

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"출력 디렉토리가 생성되었습니다: {output_dir}")
        except OSError as e:
            logging.error(f"출력 디렉토리 생성 실패: {e}")
            return

    try:
        config = Config(
            input_path=args.input_path, output_path=args.output_path,
            source_tokenizer_type=args.source_tokenizer, target_tokenizer_type=args.target_tokenizer,
            embedder_type=args.embedder, # aligner_type="strict", ← 삭제
            use_parallel=args.parallel, num_workers=args.workers,
            chunk_size=args.chunk_size, verbose=args.verbose
            # source_tokenizer_config, target_tokenizer_config, embedder_config, aligner_config는
            # Config 클래스에서 기본값 또는 __post_init__을 통해 설정됨.
            # 필요시 argparse로 추가 인자를 받아 Config에 전달 가능.
        )
        
        logging.info("처리 시작...")
        logging.info(f"입력 파일: {config.input_path}")
        logging.info(f"출력 파일: {config.output_path}")
        logging.info(f"원문 토크나이저: {config.source_tokenizer_type}")
        logging.info(f"번역문 토크나이저: {config.target_tokenizer_type}")
        logging.info(f"임베더: {config.embedder_type} (캐시: {config.embedder_config.get('cache_dir', '미설정')})")
        # logging.info(f"정렬기: {config.aligner_type}") ← 삭제
        logging.info(f"병렬 처리: {'활성화' if config.use_parallel else '비활성화'} (워커: {config.num_workers}, 청크 크기: {config.chunk_size})")

        run_processing(config)
        logging.info("모든 처리 완료.")

    except ValueError as ve:
        logging.error(f"설정 오류: {ve}", exc_info=True)
    except Exception as e:
        logging.error(f"예상치 못한 에러 발생: {e}", exc_info=True)

if __name__ == "__main__":
    # if os.name == 'nt': # Windows에서 멀티프로세싱 사용 시
    #     multiprocessing.freeze_support()
    main()