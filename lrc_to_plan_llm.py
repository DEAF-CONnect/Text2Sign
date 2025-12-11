#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lrc_to_plan_llm.py

- LRC 형식 txt 파일([mm:ss.xx] 가사)을 읽어서
  각 줄을 LLM에 보내 "수어 gloss 시퀀스" plan을 만든다.
- 각 줄은 LRC 타임스탬프 기준으로 duration이 정해져 있고,
  LLM이 준 seconds 합을 이 duration에 맞게 1차 스케일링한다.
- 그 후, test_text2sign.py 의 sanitize_plan 으로
  기능어/매핑 정리를 한 뒤,
  최종적으로 전체 seconds 합을 LRC 기반 전체 길이에 맞게 2차 스케일링한다.

사용 예시:
    python3 lrc_to_plan_llm.py lyrics.txt --out 고민_1절_plan.json

그 다음:
    python3 test_text2sign.py --plan_json 고민_1절_plan.json --out 고민_1절_out.json
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any

from test_text2sign import (
    MANIFEST_CONFIG,
    TemplateStore,
    sanitize_plan,
    call_llm_make_plan,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_FPS,
)

# [mm:ss.xx] 형태
TIMESTAMP_RE = re.compile(r"\[(\d+):(\d+\.\d+)\]")


def parse_lrc(path: str) -> List[Dict[str, Any]]:
    """
    LRC(txt) 파일에서 (start_sec, text) 리스트로 파싱.
    text가 비어 있어도 '경계선'으로 유지.
    """
    lines: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            m = TIMESTAMP_RE.match(raw)
            if not m:
                continue

            mm = int(m.group(1))
            ss = float(m.group(2))
            start_sec = mm * 60 + ss

            text = TIMESTAMP_RE.sub("", raw).strip()
            lines.append({"start": start_sec, "text": text})

    lines.sort(key=lambda x: x["start"])
    return lines


def build_plan_with_llm_and_lrc(
    path: str,
    default_last_duration: float = 2.0,
) -> Dict[str, Any]:
    """
    LRC 한 줄마다:
      - LLM으로 gloss plan 생성
      - 해당 줄 duration에 맞게 seconds 재조정 (1차 스케일)
    특징:
      - 가사 없는 빈 줄의 duration은 "다음 가사 줄"에 합쳐서 사용
      - 마지막 줄 이후 default_last_duration 초를 줄 수 있음
      - sanitize_plan 이후에는 전체 seconds 합을 LRC 전체 길이에 맞게
        다시 한 번 스케일링(2차 스케일)해서 총 길이(1절 길이)를 맞춘다.
    """
    lrc_lines = parse_lrc(path)
    if not lrc_lines:
        return {"tokens": []}

    # 1) 각 줄 duration 계산
    for i in range(len(lrc_lines)):
        start = lrc_lines[i]["start"]
        if i + 1 < len(lrc_lines):
            end = lrc_lines[i + 1]["start"]
        else:
            end = start + default_last_duration
        duration = max(end - start, 0.0)
        lrc_lines[i]["duration"] = duration

    # 2) TemplateStore 로딩 → 허용 gloss 목록
    print("[TemplateStore] 여러 manifest 로딩 중...")
    ts = TemplateStore(MANIFEST_CONFIG, fps=DEFAULT_FPS)
    allowed_glosses = ts.get_all_glosses()
    print(f"[TemplateStore] 허용 gloss 개수: {len(allowed_glosses)}")

    all_tokens: List[Dict[str, Any]] = []
    carry = 0.0  # 앞에서 모인 빈 줄 duration

    # 🔹 LRC 기반 목표 전체 시간(초)
    total_target_sec = 0.0

    for i, line in enumerate(lrc_lines):
        text = line["text"]
        dur = float(line.get("duration", 0.0))

        # 빈 줄이면 duration만 carry에 모아두고 토큰 생성 X
        if not text:
            carry += dur
            print(
                f"[LRC] {line['start']:.2f}s 빈 줄, duration={dur:.2f}s "
                f"(누적 carry={carry:.2f}s)"
            )
            continue

        # 이 줄의 실제 사용 duration = 자기 duration + 앞에서 모인 빈 줄 시간
        effective_duration = dur + carry
        carry = 0.0

        # 최소 0.3초는 보장
        effective_duration = max(effective_duration, 0.3)

        # 전체 목표 시간에 이 줄 duration 합산
        total_target_sec += effective_duration

        print(
            f"[LRC] {line['start']:.2f}s \"{text}\" "
            f"duration={dur:.2f}s, effective={effective_duration:.2f}s"
        )

        # 3) 이 줄 전체를 LLM에 넘겨서 gloss plan 생성
        raw_plan = call_llm_make_plan(
            text=text,
            allowed_glosses=allowed_glosses,
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        line_tokens = raw_plan.get("tokens") or []
        if not line_tokens:
            print("  [WARN] LLM이 이 줄에 대해 토큰을 생성하지 않았습니다. 건너뜀.")
            continue

        # 4) LLM이 준 seconds를 effective_duration에 맞게 1차 스케일링
        total_llm_sec = 0.0
        for t in line_tokens:
            try:
                total_llm_sec += float(t.get("seconds", 1.0))
            except Exception:
                total_llm_sec += 1.0

        if total_llm_sec <= 0:
            total_llm_sec = float(len(line_tokens)) or 1.0

        scale = effective_duration / total_llm_sec
        print(
            f"  [LLM] 토큰 {len(line_tokens)}개, LLM seconds 합={total_llm_sec:.2f}, "
            f"scale={scale:.3f}"
        )

        for t in line_tokens:
            try:
                base_sec = float(t.get("seconds", 1.0))
            except Exception:
                base_sec = 1.0
            sec = base_sec * scale
            # 너무 짧지 않게 바닥 0.2초
            t["seconds"] = round(max(sec, 0.2), 3)

        all_tokens.extend(line_tokens)

    raw_plan_all = {"tokens": all_tokens}
    print(f"[plan] LLM+LRC 병합 후 raw tokens: {len(all_tokens)}개")

    # 5) 최종 sanitize (기능어 제거 + 허용 gloss 매핑)
    final_plan = sanitize_plan(raw_plan_all, allowed_glosses)
    tokens = final_plan.get("tokens") or []
    print(f"[plan] sanitize_plan 이후 tokens: {len(tokens)}개")

    # 6) 최종 전체 길이를 LRC 기반 total_target_sec에 맞게 재스케일 (2차)
    sanitized_total = 0.0
    for t in tokens:
        try:
            sanitized_total += float(t.get("seconds", 0.0))
        except Exception:
            pass

    print(
        f"[plan] LRC target total={total_target_sec:.3f}s, "
        f"sanitized total={sanitized_total:.3f}s"
    )

    if sanitized_total > 0 and total_target_sec > 0:
        scale_final = total_target_sec / sanitized_total
        print(f"[plan] 최종 seconds 스케일 팩터: {scale_final:.3f}")

        for t in tokens:
            try:
                sec = float(t.get("seconds", 0.0))
            except Exception:
                sec = 0.0
            new_sec = max(0.2, sec * scale_final)  # 최소 0.2초 유지
            t["seconds"] = round(new_sec, 3)

    return {"tokens": tokens}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("lrc_path", help="입력 LRC/txt 파일 경로")
    parser.add_argument(
        "--out",
        default="plan_from_lrc_llm.json",
        help="출력 plan.json 경로 (default: plan_from_lrc_llm.json)",
    )
    args = parser.parse_args()

    lrc_path = Path(args.lrc_path)
    if not lrc_path.is_file():
        raise SystemExit(f"입력 파일을 찾을 수 없습니다: {lrc_path}")

    plan = build_plan_with_llm_and_lrc(str(lrc_path))
    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OUT] plan 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
