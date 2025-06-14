# main.py - 정렬 파이프라인 실행 진입점 (CLI 인터페이스 및 로깅 설정 포함)
# 패키지 전체 파이프라인 실행 예시: split_root 디렉토리에서
#   python -m aligner_pipeline.main input.xlsx output.xlsx

import os
import sys
import logging
import argparse
from io_manager import process_file

def main():
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARNING,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="한문-현대어 구 분할 파이프라인")
    parser.add_argument("input_file", help="입력 Excel 파일 (.xls/.xlsx)")
    parser.add_argument("output_file", help="출력 Excel 파일 (.xlsx)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose 모드: INFO 레벨 로그까지 출력"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        logger.error(f"입력 파일이 존재하지 않습니다: {args.input_file}")
        sys.exit(1)
    if not args.input_file.lower().endswith((".xls", ".xlsx")):
        logger.error("첫 번째 인자는 .xls/.xlsx 확장자여야 합니다.")
        sys.exit(1)
    if not args.output_file.lower().endswith(".xlsx"):
        logger.error("두 번째 인자는 .xlsx 확장자여야 합니다.")
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Verbose 모드 활성화됨: INFO 레벨 로그를 출력합니다.")

    try:
        process_file(  # 수정된 부분
            input_path=args.input_file,
            output_path=args.output_file,
            batch_size=128,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"파이프라인 실행 중 치명적 오류 발생: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
