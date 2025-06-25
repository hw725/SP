from sa_tokenizers import split_src_meaning_units, split_tgt_meaning_units

# 실제 테스트할 src/tgt 예시 (여기에 원하는 문장 입력)
src = "예시 원문을 여기에 입력하세요"
tgt = "예시 번역문을 여기에 입력하세요"

print("[SRC]", src)
print("[TGT]", tgt)
print("[SRC_UNITS]", split_src_meaning_units(src))
print("[TGT_UNITS]", split_tgt_meaning_units(src, tgt, use_semantic=False))
