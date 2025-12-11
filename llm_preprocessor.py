#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 전처리 → {"tokens":[{"gloss":".","seconds":.},]} plan.json 생성
(옵션) generator를 --plan_json 으로 이어서 최종 JSONL까지 출력

요구:
  - pip install openai
  - 환경변수 OPENROUTER_API_KEY 설정 (OpenRouter API 키)
  - templates_root(들) 하위에 <글로스> 디렉터리와 *.npz 가 존재해야 함
  - text2sign_retrieval_full.py 에 --plan_json 옵션이 구현되어 있거나
    (미구현이면 이 스크립트로 plan만 만들고, generator는 기존 방식으로 텍스트 입력 사용)

사용 예:
  # 1) 계획만 만들기 (디렉토리 스캔 기반)
  python llm_preprocessor.py --templates_root "data/templates_crowd_v2" \
      --text "오늘 공연 와줘서 감사합니다" --out_plan "out/plan.json"

  # 2) manifest 기반으로 AllowedGlosses 사용
  python llm_preprocessor.py --templates_root "data/templates_crowd_v3" \
      --manifest_json "manifest_crowd_v3.json" \
      --text "오늘 공연 와줘서 감사합니다" --out_plan "out/plan.json"

  # 3) 계획 만들고 곧바로 generator 실행(권장)
  python llm_preprocessor.py --templates_root "data/templates_crowd_v2" \
      --text "오늘 공연 와줘서 감사합니다" --out_plan "out/plan.json" \
      --generate --generator_path "text2sign_retrieval_full.py" \
      --out_jsonl "out/sign_sentence.jsonl" --fps 30 --fade_frames 6 --gap_frames 3 --pick median
"""

import os, re, json, time, argparse, subprocess
from pathlib import Path
from typing import List, Dict, Any, Set

# ----------------------------
# AllowedGlosses 수집
# ----------------------------
def build_allowed_glosses(templates_root: str) -> List[str]:
    """
    templates_root: "rootA;rootB" 처럼 ; 로 여러 경로 전달 가능
    gloss 후보는 각 루트 하위의 '글로스명' 디렉터리(그 안에 *.npz 존재)에서 수집
    """
    roots = [p.strip() for p in templates_root.split(";") if p.strip()]
    gls: Set[str] = set()
    for rt in roots:
        rt_path = Path(rt)
        if not rt_path.exists():
            continue
        # 글로스 폴더: 그 안에 npz가 하나 이상 있는 디렉토리
        for d in rt_path.rglob("*"):
            if d.is_dir():
                try:
                    if any(pp.suffix == ".npz" for pp in d.glob("*.npz")):
                        gls.add(d.name)
                except Exception:
                    pass
    return sorted(gls)


def build_allowed_glosses_from_manifest(manifest_path: str) -> List[str]:
    """
    manifest_crowd_v3.json 처럼 meta/index 구조를 가진 manifest에서
    사용 가능한 글로스 목록을 뽑아 AllowedGlosses 로 사용한다.

    지원 형태:
      - {"meta": {...}, "index": { "<gloss>": {...}, ... }}
      - 또는 {"<gloss>": {...}, ...} 처럼 바로 매핑된 형태
    """
    p = Path(manifest_path)
    if not p.exists():
        raise RuntimeError(f"manifest 파일을 찾을 수 없습니다: {manifest_path}")

    with p.open("r", encoding="utf-8") as f:
        mani = json.load(f)

    # manifest_crowd_v3.json 구조 (meta + index) 지원 :contentReference[oaicite:0]{index=0}
    if isinstance(mani, dict) and "index" in mani:
        index = mani.get("index", {})
    else:
        # 이미 gloss → entry 매핑 구조인 경우
        index = mani

    if not isinstance(index, dict):
        raise RuntimeError(f"manifest 형식을 이해할 수 없습니다: {manifest_path}")

    glosses = sorted(index.keys())
    if not glosses:
        raise RuntimeError(f"manifest에서 글로스를 찾지 못했습니다: {manifest_path}")

    return glosses

# ----------------------------
# OpenRouter LLM 호출 (OpenAI SDK 호환)
# ----------------------------
def call_llm_make_plan(
    text: str,
    allowed: List[str],
    model: str = "openrouter/auto",
    temperature: float = 0.1,
    max_tokens: int = 512
) -> Dict[str, Any]:
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENROUTER_API_KEY 가 없습니다. OpenRouter API 키를 설정하세요.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    allowed_preview = ", ".join(allowed[:200])  # 프롬프트 길이 보호
    system = f"""
당신은 한국어 문장을 수어 글로스 시퀀스로 변환하는 전처리기입니다.
반드시 아래 JSON만 출력하세요. 설명, 코드블록, 마크다운 금지.

규칙:
- 글로스는 AllowedGlosses 집합 안에서만 선택합니다.
- 문장부호(, . ! ? 등)는 <PAUSE>로 변환해도 됩니다.
- seconds는 0.3~2.5 범위 소수로 추정(기본 1.0). 중요 토큰은 1.2~1.6까지 늘릴 수 있음.
- AllowedGlosses에 정확히 없으면 가장 의미가 가까운 글로스 1개로 치환합니다.
- 출력 스키마 외 필드는 절대 넣지 말 것.

AllowedGlosses(일부): {allowed_preview}
출력 스키마:
{{"tokens":[{{"gloss":"...", "seconds":1.0}}, ...]}}
""".strip()

    for attempt in range(3):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
        )
        raw = resp.choices[0].message.content.strip()
        try:
            plan = json.loads(raw)
            if not isinstance(plan, dict) or "tokens" not in plan:
                raise ValueError("LLM 응답에 tokens 필드가 없습니다.")
            return plan
        except Exception:
            if attempt == 2:
                raise RuntimeError(f"LLM JSON 파싱 실패: {raw[:300]}")
            time.sleep(0.4)
    raise RuntimeError("LLM 호출 실패")

# ----------------------------
# 계획 검증/정규화
# ----------------------------
_PUNCT_RE = re.compile(r"[,\.\!\?\:\;\(\)\[\]\{\}…~\-_/]")

def sanitize_plan(plan: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
    allowed_set = set(allowed)
    tokens = []
    for t in plan.get("tokens", []):
        g = str(t.get("gloss", "")).strip()
        s = float(t.get("seconds", 1.0))
        # 문장부호 → <PAUSE>
        if _PUNCT_RE.fullmatch(g):
            g = "<PAUSE>"
        # 글로스 범위 보정
        if g != "<PAUSE>" and g not in allowed_set:
            # 가장 가까운 후보 찾기(간단 edit distance)
            g = _nearest_gloss(g, allowed) or g
        # seconds 범위 보정
        s = max(0.3, min(2.5, s))
        tokens.append({"gloss": g, "seconds": s})
    if not tokens:
        tokens = [{"gloss": "<PAUSE>", "seconds": 0.5}]
    return {"tokens": tokens}

def _edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return dp[-1]

def _nearest_gloss(name: str, allowed: List[str], max_ed: int = 2) -> str:
    if name in allowed:
        return name
    best, best_ed = None, 10**9
    for g in allowed:
        ed = _edit_distance(name, g)
        if ed < best_ed:
            best_ed, best = ed, g
    return best if best_ed <= max_ed else name

# ----------------------------
# 파일 입출력 / generator 연동
# ----------------------------
def write_plan(plan: Dict[str, Any], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False)
    print(f"✅ plan.json 저장: {out_path}")

def run_generator_with_plan(
    generator_path: str,
    templates_root: str,
    plan_json: str,
    out_jsonl: str,
    fps: int,
    fade_frames: int,
    gap_frames: int,
    pick: str
):
    cmd = [
        "python", generator_path,
        "--templates_root", templates_root,
        "--plan_json", plan_json,
        "--out", out_jsonl,
        "--fps", str(fps),
        "--fade_frames", str(fade_frames),
        "--gap_frames", str(gap_frames),
        "--pick", pick,
    ]
    print("▶ 실행:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"🎬 generator 완료 → {out_jsonl}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_root", required=True, help="템플릿 루트(여러 개는 ; 로 연결)")
    ap.add_argument("--text", required=True, help="입력 문장 (STT 결과)")
    ap.add_argument("--out_plan", default="out/plan.json", help="LLM 계획 JSON 저장 경로")
    ap.add_argument("--model", default="openrouter/auto", help="OpenRouter 모델 ID")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_tokens", type=int, default=512)

    # manifest 기반 AllowedGlosses (옵션)
    ap.add_argument(
        "--manifest_json",
        help="manifest_crowd_v3.json 같은 템플릿 인덱스 JSON. 주어지면 templates_root 디렉토리 스캔 대신 여기의 글로스 목록만 AllowedGlosses로 사용."
    )

    # generator 연동(옵션)
    ap.add_argument("--generate", action="store_true", help="계획 생성 후 generator 실행")
    ap.add_argument("--generator_path", default="text2sign_retrieval_full.py")
    ap.add_argument("--out_jsonl", default="out/sign_sentence.jsonl")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade_frames", type=int, default=6)
    ap.add_argument("--gap_frames", type=int, default=3)
    ap.add_argument("--pick", choices=["median","random"], default="median")

    args = ap.parse_args()

    # 1) 허용 글로스 수집: manifest 우선, 없으면 디렉토리 스캔 
    if args.manifest_json:
        allowed = build_allowed_glosses_from_manifest(args.manifest_json)
    else:
        allowed = build_allowed_glosses(args.templates_root)

    if not allowed:
        if args.manifest_json:
            raise RuntimeError(f"manifest에 글로스가 없습니다: {args.manifest_json}")
        else:
            raise RuntimeError(f"템플릿이 비어있습니다: {args.templates_root}")

    # 2) LLM 호출 → plan 초안
    raw_plan = call_llm_make_plan(
        args.text,
        allowed,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # 3) 서버단 정규화
    plan = sanitize_plan(raw_plan, allowed)

    # 4) 저장
    write_plan(plan, args.out_plan)

    # 5) (옵션) generator 실행
    if args.generate:
        run_generator_with_plan(
            generator_path=args.generator_path,
            templates_root=args.templates_root,
            plan_json=args.out_plan,
            out_jsonl=args.out_jsonl,
            fps=args.fps,
            fade_frames=args.fade_frames,
            gap_frames=args.gap_frames,
            pick=args.pick,
        )

if __name__ == "__main__":
    main()
