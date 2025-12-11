#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realtime_sign_llm_server.py

- STT에서 전달된 한국어 문장을 받아서
  1) LLM으로 gloss plan 생성
  2) plan 정리(조사/어미 제거, 허용되지 않은 gloss 매핑)
  3) 템플릿 npz에서 프레임 생성
  4) WebSocket으로 Unity에 실시간 전송

특징:
- 템플릿 npz 경로는 manifest.json에서만 읽는다.
- TemplateStore는 "실제로 npz가 존재하는 gloss"만 템플릿으로 사용한다.
- allowed_glosses는 manifest에서 직접 읽어온 전체 gloss 목록을 사용한다.
- 조사(은/는/이/가/을/를/...) + 어미(요, 습니다, 겠, 았/었 ...)는
  plan 정리 단계에서 제거하고, 앞 토큰 duration에 합산한다.
- 허용되지 않은 gloss는 가능한 한 allowed_glosses 중 가장 비슷한 것으로 강제 매핑한다.
"""

import asyncio
import json
import os
import random
import difflib
import time
from fastapi import Request
from fastapi import Body
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ================================
# 기본 설정
# ================================

DEFAULT_FPS = 30
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# 전체 재생 속도 스케일 (1.0 = 그대로, 1.3 = 30% 느리게, 0.8 = 20% 빠르게)
GLOBAL_SPEED_SCALE = float(os.getenv("GLOBAL_SPEED_SCALE", "5.0"))

# 조사(문장 성분이 아니라 기능어 역할) 목록
PARTICLE_GLOSSES: Set[str] = {
    "은", "는", "이", "가",
    "을", "를",
    "에", "에서", "에게", "한테",
    "으로", "로",
    "와", "과", "하고", "랑",
    "도", "만", "까지", "부터",
    "이나", "나",
    "조차", "마저", "밖에",
    "처럼", "만큼", "보다",
}

# 자주 떨어져 나오는 어미/종결 표현들
ENDING_GLOSSES: Set[str] = {
    # 한 글자
    "요", "다", "네", "군", "지", "고", "게",
    # 두 글자 이상 (종결/경어/시제 느낌들)
    "네요", "군요", "구나", "구요",
    "습니다", "습니까", "습니다만",
    "겠", "겠어요", "겠습",
    "었", "았", "었어요", "았어요",
    "더라", "더니", "더군요",
}

# 조사 + 어미 = 기능어 gloss 모음
FUNCTION_GLOSSES: Set[str] = PARTICLE_GLOSSES | ENDING_GLOSSES

# manifest / root 설정 (환경변수로 덮어쓰기 가능)
MANIFEST_CONFIG: List[Dict[str, str]] = [
    {
        "manifest": os.getenv(
            "MANIFEST_CROWD", "/YOUR_SERVER_PATH/manifest_crowd_v3_small.json"
        ),
        "root": os.getenv(
            "ROOT_CROWD", "/YOUR_SERVER_PATH/data/templates_crowd_v3"
        ),
    },
    {
        "manifest": os.getenv(
            "MANIFEST_REALSEN", "/YOUR_SERVER_PATH/manifest_realsen_v3_small.json"
        ),
        "root": os.getenv(
            "ROOT_REALSEN", "/YOUR_SERVER_PATH/data/templates_realsen_v3"
        ),
    },
    {
        "manifest": os.getenv(
            "MANIFEST_REALWORD", "/YOUR_SERVER_PATH/manifest_realword_v3_small.json"
        ),
        "root": os.getenv(
            "ROOT_REALWORD", "/YOUR_SERVER_PATH/data/templates_realword_v3"
        ),
    },
]

# ================================
# 디버그 출력용 폴더 (STT + Unity로 갈 데이터 저장)
# ================================
DEBUG_OUT_DIR = Path(os.getenv("DEBUG_OUT_DIR", "debug_out"))
DEBUG_OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[debug] debug_out 디렉토리: {DEBUG_OUT_DIR.resolve()}")


# ================================
# OpenAI client
# ================================

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ================================
# TemplateStore
# ================================

@dataclass
class TemplateEntry:
    gloss: str
    path: Path

def resample_kpt_for_duration(kpt, target_seconds, fps):
    T = kpt.shape[0]
    if T <= 0:
        return kpt

    # 🔹 LLM이 준 seconds에 전체 배율 곱하기
    scaled = float(target_seconds) * GLOBAL_SPEED_SCALE

    # 너무 극단적이지 않게 범위 제한
    scaled = max(0.3, min(3.5, scaled))  # 3.5초까지 허용 (원하면 조절)

    target_T = int(round(scaled * fps))
    if target_T <= 0:
        target_T = 1

    if abs(target_T - T) < 2:
        return kpt

    idx = np.linspace(0, T - 1, target_T).astype(int)
    idx = np.clip(idx, 0, T - 1)
    return kpt[idx]



class TemplateStore:
    """
    manifest.json 여러 개에서 gloss -> [npz 후보들] 매핑을 만들고,
    gloss에 대한 out.json 스타일 프레임 리스트를 생성한다.

    특징:
    - 템플릿 폴더 전체를 os.walk / glob 으로 탐색하지 않는다.
    - gloss 목록과 npz 경로는 오직 manifest.json에서만 읽는다.
    - manifest에만 있고 실제 npz가 없는 샘플은 로딩 단계에서 스킵한다.
    - 따라서 TemplateStore 내부에는 "실제로 npz가 최소 1개 있는 gloss"만 존재한다.
    """

    def __init__(self, configs: List[Dict[str, str]], fps: int = DEFAULT_FPS) -> None:
        self.fps = fps
        self._entries_by_gloss: Dict[str, List[TemplateEntry]] = {}
        self._load_manifests(configs)

    def _load_manifests(self, configs: List[Dict[str, str]]) -> None:
        total = 0
        skipped = 0

        for cfg in configs:
            mpath = Path(cfg["manifest"])
            root = Path(cfg["root"])
            if not mpath.is_file():
                print(f"[TemplateStore] manifest 없음: {mpath}")
                continue

            try:
                data = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[TemplateStore] manifest 로딩 실패: {mpath} ({e})")
                continue

            # 1) crowd/realsen 스타일: index 기반 구조
            index = data.get("index")
            if isinstance(index, dict):
                for gloss, info in index.items():
                    samples = info.get("samples") or []
                    if not isinstance(samples, list):
                        continue
                    for s in samples:
                        rel = s.get("path")
                        if not rel:
                            continue

                        rel_posix = rel.replace("\\", "/")
                        full = (root / rel_posix).resolve()

                        # 실제 파일 있는 것만 등록
                        if not full.is_file():
                            skipped += 1
                            continue

                        self._entries_by_gloss.setdefault(gloss, []).append(
                            TemplateEntry(gloss=gloss, path=full)
                        )
                        total += 1
            else:
                # 2) fallback: entries/templates 리스트 구조
                entries = data.get("entries") or data.get("templates") or []
                if isinstance(entries, list):
                    for e in entries:
                        gloss = e.get("gloss")
                        if not gloss:
                            continue
                        rel = (
                            e.get("npz_path")
                            or e.get("relpath")
                            or e.get("path")
                            or e.get("file")
                        )
                        if not rel:
                            continue

                        rel_posix = str(rel).replace("\\", "/")
                        full = (root / rel_posix).resolve()

                        if not full.is_file():
                            skipped += 1
                            continue

                        self._entries_by_gloss.setdefault(gloss, []).append(
                            TemplateEntry(gloss=gloss, path=full)
                        )
                        total += 1
                else:
                    print(f"[TemplateStore] index/entries 없음: {mpath}")

        # 실제 npz가 하나도 없는 gloss는 제거
        empty_glosses = [g for g, arr in self._entries_by_gloss.items() if not arr]
        for g in empty_glosses:
            self._entries_by_gloss.pop(g, None)

        print(
            f"[TemplateStore] load ok: gloss {len(self._entries_by_gloss)}개, "
            f"entries {total}개, skipped_missing_files={skipped}"
        )

    def get_all_glosses(self) -> List[str]:
        """현재 TemplateStore에 실제 npz가 있는 gloss 목록"""
        return sorted(self._entries_by_gloss.keys())

    def _pick_entry_for_gloss(self, gloss: str) -> Optional[TemplateEntry]:
        arr = self._entries_by_gloss.get(gloss)
        if not arr:
            return None
        return random.choice(arr)

    def load_kpt_for_gloss(self, gloss: str) -> Optional[np.ndarray]:
        ent = self._pick_entry_for_gloss(gloss)
        if not ent:
            print(f"[TemplateStore] gloss='{gloss}' 템플릿 없음")
            return None
        if not ent.path.is_file():
            print(f"[TemplateStore] npz 파일 없음(런타임): {ent.path}")
            return None
        try:
            data = np.load(str(ent.path), allow_pickle=True)
            if "kpt" not in data.files:
                print(f"[TemplateStore] 'kpt' 없음: {ent.path}")
                return None
            return data["kpt"]  # (T, 67, D)
        except Exception as e:
            print(f"[TemplateStore] npz 로딩 실패: {ent.path} ({e})")
            return None

    def make_outjson_frames_for_gloss(
        self,
        gloss: str,
        seconds: float,
        *,
        fps: int,
        t_start_ms: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        seconds 값에 맞춰 npz 프레임을 resample 후 Unity 전송용 프레임 리스트 생성
        """

        # --- 1) 템플릿 로드 ---
        kpt = self.load_kpt_for_gloss(gloss)
        if kpt is None:
            return None

        # --- 2) seconds 기반 resample ---
        fps = max(fps, 1)
        kpt = resample_kpt_for_duration(kpt, seconds, fps)
        T = kpt.shape[0]
        if T <= 0:
            return None

        # --- 3) 시간 설정 ---
        step_ms = int(1000 / fps)
        t_ms = t_start_ms

        frames: List[Dict[str, Any]] = []

        # --- 4) 프레임 생성 ---
        for i in range(T):
            pts = kpt[i]  # shape = (67, D)

            if pts.shape[0] < 67:
                print(f"[TemplateStore] 포인트 수 67 미만: gloss={gloss}")
                return None

            pose_list = []
            left_list = []
            right_list = []

            # pose (0~24)
            for j in range(25):
                pose_list.append({
                    "x": float(pts[j, 0]),
                    "y": float(pts[j, 1]),
                    "visibility": 1.0
                })

            # left_hand (25~45)
            for j in range(25, 46):
                left_list.append({
                    "x": float(pts[j, 0]),
                    "y": float(pts[j, 1]),
                    "visibility": 1.0
                })

            # right_hand (46~66)
            for j in range(46, 67):
                right_list.append({
                    "x": float(pts[j, 0]),
                    "y": float(pts[j, 1]),
                    "visibility": 1.0
                })

            frames.append({
                "t": t_ms,
                "pose": pose_list,
                "left_hand": left_list,
                "right_hand": right_list,
                "token": gloss
            })

            t_ms += step_ms

        return frames



# ================================
# manifest에서 allowed_glosses 읽기
# (npz 존재 여부와 상관없이 manifest에 등장하는 모든 gloss)
# ================================

def load_allowed_glosses_from_manifest(configs: List[Dict[str, str]]) -> List[str]:
    gloss_set: Set[str] = set()

    for cfg in configs:
        mpath = Path(cfg["manifest"])
        if not mpath.is_file():
            print(f"[allowed_glosses] manifest 없음: {mpath}")
            continue

        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[allowed_glosses] manifest 로딩 실패: {mpath} ({e})")
            continue

        index = data.get("index")
        if isinstance(index, dict):
            for gloss in index.keys():
                if isinstance(gloss, str):
                    gloss_set.add(gloss)
        else:
            entries = data.get("entries") or data.get("templates") or []
            if isinstance(entries, list):
                for e in entries:
                    gloss = e.get("gloss")
                    if isinstance(gloss, str):
                        gloss_set.add(gloss)

    gloss_list = sorted(gloss_set)
    print(f"[allowed_glosses] manifest 기반 gloss 수: {len(gloss_list)}")
    return gloss_list


# ================================
# 의미 기반 gloss 매핑 (LLM)
# ================================

_SEM_CACHE: Dict[str, Optional[str]] = {}


def semantic_match_gloss(
    gloss: str,
    allowed_glosses: List[str],
    *,
    model: str = DEFAULT_LLM_MODEL,
) -> Optional[str]:
    """
    허용된 gloss 목록(= manifest에서 읽어온 전체 gloss들) 중에서
    주어진 gloss 또는 문장과 '의미적으로' 가장 비슷한 것을 LLM에게 골라달라고 요청.
    - 반드시 allowed_glosses 중 하나만 고르게 강제
    - 적당한 게 없으면 "NONE"을 반환하게 하고, 그 경우 None 리턴
    - 같은 입력에 대한 호출은 캐시에 저장해서 중복 호출 방지
    """
    global _SEM_CACHE

    if gloss in _SEM_CACHE:
        return _SEM_CACHE[gloss]

    if not allowed_glosses:
        return None

    gloss_list_for_prompt = ", ".join(allowed_glosses[:300])

    system_msg = (
        "You are choosing the most semantically similar KSL gloss.\n"
        "You MUST respond with exactly one token: either one of the allowed glosses or 'NONE'.\n"
    )

    user_msg = (
        f"TARGET: {gloss}\n\n"
        f"ALLOWED_GLOSSES:\n{gloss_list_for_prompt}\n\n"
        "Choose ONE gloss from ALLOWED_GLOSSES that is most semantically related to TARGET.\n"
        "If none is appropriate, answer with NONE.\n"
    )

    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        print(f"[semantic] LLM 호출 실패: {e}")
        _SEM_CACHE[gloss] = None
        return None

    result = (resp.choices[0].message.content or "").strip()
    result = result.strip("\"'")

    if result.upper() == "NONE":
        print(f"[semantic] 의미 매핑 실패: {gloss}")
        _SEM_CACHE[gloss] = None
        return None

    if result not in allowed_glosses:
        print(f"[semantic] LLM 결과가 allowed_glosses에 없음: {result}")
        _SEM_CACHE[gloss] = None
        return None

    print(f"[semantic] {gloss} -> {result}")
    _SEM_CACHE[gloss] = result
    return result


# ================================
# LLM plan 생성
# ================================

def call_llm_make_plan(
    text: str,
    allowed_glosses: List[str],
    *,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    입력 텍스트와 허용 gloss 목록을 기반으로
    {"tokens":[{"gloss","seconds"}...], "meta":{...}} JSON을 반환.
    - LLM이 tokens를 비워서 보내더라도, 여기서 최소 1개 토큰을 강제로 생성한다.
    """
    if not allowed_glosses:
        raise RuntimeError("allowed_glosses가 비어 있습니다. manifest 로딩을 확인하세요.")

    gloss_list_for_prompt = ", ".join(allowed_glosses)

    system_msg = (
        "You are a planning module for a Text-to-KSL system.\n"
        "Your job is to map the Korean input text into a sequence of gloss tokens.\n"
        "You MUST return a JSON object with keys 'tokens' and 'meta'.\n"
        "CRITICAL RULES:\n"
        "- Every 'gloss' MUST be chosen from the allowed gloss list.\n"
        "- You MUST NEVER return an empty 'tokens' list.\n"
        "- If you are unsure, choose the closest gloss and still output at least one token.\n"
        "- For Korean function words/particles (조사/어미) such as 에, 에서, 으로, 로, 와, 과, 도, 만, 까지, 부터, "
        "에게, 한테, 나, 이나, 랑, 하고, 요, 다, 네, etc.,\n"
        "  you should usually NOT output them as separate gloss tokens. Prefer content words "
        "(nouns, verbs, adjectives, adverbs).\n"
        "- Each token MUST have a 'seconds' field.\n"
        "- 'seconds' means how long this gloss should be signed.\n"
        "- Use different durations per token, do NOT make all seconds the same.\n"
        "- Typical range for seconds is 0.6 ~ 1.8 seconds.\n"
        "- Short interjections (e.g., greetings, small ad-libs) → around 0.6~1.0 seconds.\n"
        "- Important content words (verbs, adjectives) → around 0.9~1.5 seconds.\n"
        "- Very long or emphasized actions can go up to 1.8 seconds, but rarely.\n"
    )

    user_msg = (
        f"INPUT_TEXT: {text}\n\n"
        f"ALLOWED_GLOSSES (full list, {len(allowed_glosses)} items):\n"
        f"{gloss_list_for_prompt}\n\n"
        "Return JSON like:\n"
        "{\n"
        '  \"tokens\": [\n'
        '    {\"gloss\": \"안녕하세요\", \"seconds\": 0.9},\n'
        '    {\"gloss\": \"감사\", \"seconds\": 1.2}\n'
        "  ],\n"
        '  \"meta\": {\"reason\": \"...\"}\n'
        "}\n"
    )

    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("LLM이 빈 응답을 반환했습니다.")

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM JSON 파싱 실패: {e} / raw={raw[:200]}") from e

    #print("[DEBUG] raw plan from LLM:", plan)

    # LLM이 tokens를 비웠을 경우, 여기서 강제로 1개 토큰 생성
    tokens = plan.get("tokens")
    if not isinstance(tokens, list) or len(tokens) == 0:
        print("[WARN] LLM이 빈 tokens를 반환했습니다. 강제로 토큰을 생성합니다.")

        mapped = semantic_match_gloss(text, allowed_glosses, model=model)
        if not mapped:
            mapped = allowed_glosses[0]
            print(f"[fallback] semantic 매핑 실패 → allowed_glosses[0] 사용: {mapped}")

        plan["tokens"] = [{"gloss": mapped, "seconds": 1.0}]
        meta = plan.get("meta") or {}
        meta["fallback_from_empty_tokens"] = True
        plan["meta"] = meta

        print("[DEBUG] fixed plan tokens:", plan["tokens"])

    return plan


# ================================
# plan 정리 (조사 + 어미 기능어 처리 + 매핑)
# ================================

def sanitize_plan(plan: Dict[str, Any], allowed_glosses: List[str]) -> Dict[str, Any]:
    """
    plan["tokens"]를 정리:
    1) 조사/어미 토큰(FUNCTION_GLOSSES)은 최종 gloss 목록에서 제거하고,
       가능한 경우 앞 토큰 duration(seconds)에 합산.
    2) 허용되지 않은 gloss는 가능한 한 allowed_glosses 중 가장 가까운 것으로 매핑:
       - 부분문자열 기반 후보
       - 의미 기반 매핑(semantic_match_gloss)
       - 문자열 유사도 기반 difflib
       - 그래도 실패하면 allowed_glosses[0] 강제 선택
    3) seconds를 [0.3, 2.5] 범위로 클램프
    """
    tokens = plan.get("tokens") or []
    if not isinstance(tokens, list):
        raise RuntimeError("plan['tokens']가 list가 아닙니다.")

    allowed_set: Set[str] = set(allowed_glosses)
    cleaned: List[Dict[str, Any]] = []

    for t in tokens:
        if not isinstance(t, dict):
            continue

        gloss = t.get("gloss")
        seconds = t.get("seconds", 1.0)

        if not isinstance(gloss, str):
            gloss = str(gloss)

        # seconds를 먼저 정규화
        try:
            sec = float(seconds)
        except Exception:
            sec = 1.0
        sec = max(0.3, min(2.5, sec))

        # 조사/어미(기능어) 토큰이면: 최종 gloss 목록에서 제거하고, 가능한 경우 앞 토큰 길이에 합산
        if gloss in FUNCTION_GLOSSES:
            if cleaned:
                cleaned[-1]["seconds"] += sec
                print(f"[plan] 기능어 gloss 스킵 + 앞 토큰 duration 합산: {gloss}")
            else:
                print(f"[plan] 기능어 gloss 스킵(앞 토큰 없음): {gloss}")
            continue

        final_gloss = gloss

        # 이미 허용된 gloss면 그대로 사용
        if gloss not in allowed_set:
            mapped: Optional[str] = None

            # (1) 부분문자열 기반 매핑 (길이 2 이상 gloss만 시도)
            if len(gloss) >= 2:
                sub_candidates = [g for g in allowed_glosses if gloss in g or g in gloss]
            else:
                sub_candidates = []

            if sub_candidates:
                sub_candidates.sort(key=len)
                mapped = sub_candidates[0]
                print(f"[plan] 부분문자열 매핑: {gloss} -> {mapped}")

            # (2) 의미 기반 매핑 (LLM)
            if not mapped:
                mapped = semantic_match_gloss(gloss, allowed_glosses)
                if mapped:
                    print(f"[plan] 의미 매핑: {gloss} -> {mapped}")

            # (3) 문자열 유사도 기반 매핑 (difflib)
            if not mapped:
                matches = difflib.get_close_matches(
                    gloss, allowed_glosses, n=1, cutoff=0.0
                )
                if matches:
                    mapped = matches[0]
                    print(f"[plan] 문자열 유사도 매핑: {gloss} -> {mapped}")

            # (4) 그래도 실패하면 무조건 allowed_glosses[0]
            if not mapped:
                mapped = allowed_glosses[0]
                print(f"[plan] 강제 매핑: {gloss} -> {mapped}")

            final_gloss = mapped

        cleaned.append({"gloss": final_gloss, "seconds": sec})
        # --- seconds 후처리: 없거나 이상하면 기본값 부여 / 클램프 ---
    tokens = plan.get("tokens")
    if isinstance(tokens, list):
        for t in tokens:
            if not isinstance(t, dict):
                continue
            gloss = t.get("gloss")
            raw_sec = t.get("seconds", None)

            # 기본값: 1.0
            sec = 1.0
            try:
                if raw_sec is not None:
                    sec = float(raw_sec)
            except Exception:
                sec = 1.0

            # 0.3 ~ 2.5 사이로 클램프
            if sec < 0.3:
                sec = 0.3
            if sec > 2.5:
                sec = 2.5

            t["seconds"] = sec


    plan["tokens"] = cleaned
    return plan


# ================================
# Unity WebSocket 관리
# ================================

class UnityConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"[ws] Unity connected (total={len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[ws] Unity disconnected (total={len(self.active_connections)})")

    async def broadcast_json(self, data: Any) -> None:
        if not self.active_connections:
            print("[ws] Unity 연결이 없어 frames를 보낼 수 없습니다.")
            return
        living = []
        for ws in list(self.active_connections):
            try:
                await ws.send_json(data)
                living.append(ws)
            except Exception as e:
                print(f"[ws] 전송 실패, 연결 제거: {e}")
        self.active_connections = set(living)


# ================================
# FastAPI 모델
# ================================

class STTTextRequest(BaseModel):
    text: str
    utterance_id: Optional[str] = None
    is_final: bool = True


class STTTextResponse(BaseModel):
    ok: bool
    text: str
    plan: Dict[str, Any]
    frame_count: int


# ================================
# FastAPI 앱 초기화
# ================================

app = FastAPI(title="Realtime Text2Sign LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

unity_manager = UnityConnectionManager()
template_store: Optional[TemplateStore] = None
allowed_glosses: List[str] = []


@app.on_event("startup")
async def on_startup() -> None:
    global template_store, allowed_glosses
    print("[startup] TemplateStore / allowed_glosses 로딩 시작...")

    # 1) TemplateStore: 실제 npz가 존재하는 템플릿만 로드
    template_store = TemplateStore(MANIFEST_CONFIG, fps=DEFAULT_FPS)

    # 2) allowed_glosses: manifest에 등장하는 모든 gloss (npz 존재 여부와 무관)
    allowed_glosses = load_allowed_glosses_from_manifest(MANIFEST_CONFIG)

    print(f"[startup] 허용 gloss 수(allowed_glosses): {len(allowed_glosses)}")


# ================================
# 엔드포인트
# ================================

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.websocket("/ws/unity")
async def ws_unity(websocket: WebSocket) -> None:
    await unity_manager.connect(websocket)
    try:
        while True:
            # Unity에서 보낸 메시지가 필요하면 여기서 처리
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        unity_manager.disconnect(websocket)
    except Exception as e:
        print(f"[ws] 예외 발생: {e}")
        unity_manager.disconnect(websocket)

@app.post("/stt-plain", response_model=STTTextResponse)
async def stt_plain(request: Request) -> STTTextResponse:
    """
    STT가 그냥 순수 텍스트를 보내는 경우(text/plain, form, 기타) 전용 엔드포인트.
    - Content-Type 이 뭐든 상관없이 raw body를 그대로 읽어서 문자열로 해석.
    - 내부적으로는 /stt-text 파이프라인(stt_text) 재사용.
    """
    raw = await request.body()
    try:
        text = raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        text = str(raw).strip()

    print(f"[STT-PLAIN] raw={raw!r}")
    print(f"[STT-PLAIN] text={text!r}")

    if not text:
        raise HTTPException(status_code=400, detail="STT body가 비어 있습니다.")

    # 기존 /stt-text에서 쓰던 모델 재사용
    req = STTTextRequest(text=text, utterance_id=None, is_final=True)
    return await stt_text(req)


@app.post("/stt-text", response_model=STTTextResponse)
async def stt_text(req: STTTextRequest) -> STTTextResponse:
    """
    STT에서 들어온 텍스트를 받아서
    - LLM으로 gloss plan 생성
    - plan 정제 (조사/어미 제거 + 허용 gloss 매핑)
    - ***토큰(단어)별로*** 템플릿에서 프레임 생성 후
      매 토큰마다 Unity로 WebSocket 전송

    + 디버그용:
      - STT 텍스트, plan, 토큰별 프레임, 전체 프레임을 debug_out/*.json 으로 저장
    """
    global template_store, allowed_glosses

    if template_store is None or not allowed_glosses:
        raise HTTPException(status_code=500, detail="서버 초기화 오류: 템플릿 또는 gloss 목록이 없습니다.")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")

    print(f"[STT] text={text!r}, utterance_id={req.utterance_id}, is_final={req.is_final}")

    # 1) LLM으로 plan 생성
    try:
        plan = call_llm_make_plan(
            text=text,
            allowed_glosses=allowed_glosses,
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception as e:
        # 이 부분만 수정
        try:
            # repr(e)는 \u2014 처럼 이스케이프된 형태라 ASCII로 안전
            print("[ERROR] call_llm_make_plan 실패:", repr(e))
        except Exception:
            print("[ERROR] call_llm_make_plan 실패 (print 중 또 예외 발생)")
        raise HTTPException(status_code=500, detail="LLM 오류 발생")

    # 2) plan 정제 (조사/어미 처리 + 허용 gloss 매핑)
    plan = sanitize_plan(plan, allowed_glosses)
    tokens = plan.get("tokens") or []

    if not tokens:
        print("[plan] tokens가 비어 있습니다. 아무 동작도 전송하지 않습니다.")
        return STTTextResponse(
            ok=False,
            text=text,
            plan=plan,
            frame_count=0,
        )

    total_frame_count = 0
    all_frames: List[Dict[str, Any]] = []          # 전체 프레임(유니티로 가는 것과 동일 구조)
    per_token_frames: List[Dict[str, Any]] = []    # 토큰별 프레임 목록(디버그용 메타)

    # 3) 토큰(단어)별로 프레임 생성 + 전송
    for idx, t in enumerate(tokens):
        gloss = str(t.get("gloss", "")).strip()
        try:
            seconds = float(t.get("seconds", 1.0))
        except Exception:
            seconds = 1.0

        if not gloss:
            print(f"[plan] 빈 gloss 토큰 스킵: index={idx}")
            continue

        print(f"[frames] 토큰 {idx+1}/{len(tokens)}: gloss={gloss!r}, seconds={seconds}")

        # 각 단어(토큰)마다 시간 0부터 시작하도록 t_start_ms=0으로 리셋
        t_start_ms = 0

        # 템플릿에서 해당 gloss의 프레임 생성
        gloss_frames = template_store.make_outjson_frames_for_gloss(
            gloss=gloss,
            seconds=seconds,
            fps=DEFAULT_FPS,
            t_start_ms=t_start_ms,
        )

        if not gloss_frames:
            print(f"[frames] gloss={gloss!r} 에 대한 프레임을 생성하지 못했습니다. 스킵합니다.")
            continue

        frame_count_for_token = len(gloss_frames)
        total_frame_count += frame_count_for_token

        # 전체/토큰별 디버그 저장용 누적
        all_frames.extend(gloss_frames)
        per_token_frames.append(
            {
                "gloss": gloss,
                "seconds": seconds,
                "frame_count": frame_count_for_token,
                "frames": gloss_frames,
            }
        )

        print(f"[frames] gloss={gloss!r}, frames={frame_count_for_token} → Unity로 전송")

        # 이 토큰에 대한 프레임만 Unity에 바로 전송
        try:
            await unity_manager.broadcast_json(gloss_frames)
        except Exception as e:
            print(f"[ERROR] Unity broadcast 실패 (gloss={gloss!r}): {e}")

    # 4) 디버그 파일로 저장 (STT 입력 + plan + 프레임)
    if all_frames:
        ts = int(time.time())
        safe_snippet = text.replace(" ", "_")[:10] or "no_text"
        fname = DEBUG_OUT_DIR / f"{ts}_{safe_snippet}.json"

        debug_payload = {
            "meta": {
                "text": text,
                "utterance_id": req.utterance_id,
                "is_final": req.is_final,
                "created_at": ts,
                "total_frame_count": total_frame_count,
                "token_count": len(tokens),
            },
            "plan": plan,
            "tokens": tokens,
            "per_token_frames": per_token_frames,
            "all_frames_flat": all_frames,  # Unity로 전송되는 구조와 동일(flat list)
        }

        #try:
        #    with fname.open("w", encoding="utf-8") as f:
        #        json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        #    print(f"[DEBUG] STT + Unity 프레임 디버그 파일 저장: {fname}")
        #except Exception as e:
        #    print(f"[DEBUG] 디버그 파일 저장 실패: {e}")
    else:
        print("[DEBUG] 생성된 프레임이 없어 디버그 파일 저장을 건너뜁니다.")

    # 5) HTTP 응답은 여전히 "문장 단위" 요약 정보 제공
    return STTTextResponse(
        ok=(total_frame_count > 0),
        text=text,
        plan=plan,
        frame_count=total_frame_count,
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Realtime Text2Sign LLM Server",
        "websocket": "/ws/unity",
        "stt_endpoint": "/stt-text",
    }


# ================================
# 로컬 실행용
# ================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("realtime_sign_llm_server:app", host="0.0.0.0", port=port, reload=False)
