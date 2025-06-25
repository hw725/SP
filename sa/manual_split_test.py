import sys
from sa_tokenizers import split_src_meaning_units, split_tgt_meaning_units

if __name__ == "__main__":
    print("[TEST] 직접 입력 테스트")
    src = input("SRC: ").strip()
    tgt = input("TGT: ").strip()
    print("[RESULT] src_units:", split_src_meaning_units(src))
    print("[RESULT] tgt_units:", split_tgt_meaning_units(src, tgt, use_semantic=False))
