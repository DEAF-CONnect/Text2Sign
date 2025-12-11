# batch_lyrics_llm.py
import json
import time
from pathlib import Path
from llm_preprocessor import (
    build_allowed_glosses_from_manifest,
    call_llm_make_plan,
    sanitize_plan,
)

TEMPLATES_ROOT = "C:/Users/YOUR_USERNAME/Desktop/Text2Sign/data/templates_crowd_v3"
MANIFEST_JSON  = "C:/Users/YOUR_USERNAME/Desktop/Text2Sign/manifest_crowd_v3.json"
LYRICS_PATH    = "C:/Users/YOUR_USERNAME/Desktop/Text2Sign/lyrics.txt"
OUT_DIR        = Path("plans_lyrics")

MAX_RETRY_PER_LINE = 3  # 한 줄당 최대 재시도 횟수

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # manifest 경로(문자열)를 그대로 넘김
    allowed = build_allowed_glosses_from_manifest(MANIFEST_JSON)

    with open(LYRICS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            out_path = OUT_DIR / f"{i:04d}.json"

            # 이미 생성된 plan 파일이 있으면 스킵 (다시 돌릴 때 API 아끼기)
            if out_path.exists():
                print(f"[{i}] 이미 존재 → {out_path}, 스킵")
                continue

            print(f"[{i}] '{text}' LLM 호출 중...")

            success = False
            for attempt in range(1, MAX_RETRY_PER_LINE + 1):
                try:
                    raw_plan = call_llm_make_plan(text, allowed)
                    plan = sanitize_plan(raw_plan, allowed)

                    with out_path.open("w", encoding="utf-8") as out_f:
                        json.dump(plan, out_f, ensure_ascii=False)
                    print(f"  → {out_path} 저장 완료")
                    success = True
                    break
                except RuntimeError as e:
                    print(f"  !! LLM 오류 (시도 {attempt}/{MAX_RETRY_PER_LINE}): {e}")
                    time.sleep(1.0)

            if not success:
                # 이 줄은 그냥 건너뛰고 다음 줄로 진행
                print(f"  XX [경고] {i}번째 줄 실패, 건너뜀: '{text}'")

if __name__ == "__main__":
    main()
