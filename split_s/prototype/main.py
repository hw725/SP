# main.py - 정렬 파이프라인 실행 진입점 (CLI 인터페이스 및 로깅 설정 포함)

import os
import sys
import logging
import argparse

from io_manager import process_excel

def main():
    parser = argparse.ArgumentParser(description="고전 한문-현대어 병렬 정렬 파이프라인")
    parser.add_argument("input_file", help="입력 Excel 파일 (.xls/.xlsx)")
    parser.add_argument("output_file", help="출력 Excel 파일 (.xlsx)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose 모드: INFO 레벨 로그까지 출력"
    )
    args = parser.parse_args()

    # 1) 입력 파일 존재 여부
    if not os.path.exists(args.input_file):
        print(f"[Error] 입력 파일이 존재하지 않습니다: {args.input_file}")
        sys.exit(1)
    if not args.input_file.lower().endswith(('.xls', '.xlsx')):
        print("[Error] 첫 번째 인자는 .xls/.xlsx 확장자여야 합니다.")
        sys.exit(1)

    # 2) 출력 파일 확장자 체크
    if not args.output_file.lower().endswith('.xlsx'):
        print("[Error] 두 번째 인자는 .xlsx 확장자여야 합니다.")
        sys.exit(1)

    # 3) Verbose 모드 시 로깅 레벨 조정
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # 4) 전체 파이프라인 실행
    process_excel(
        input_path=args.input_file,
        output_path=args.output_file,
        batch_size=128,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()