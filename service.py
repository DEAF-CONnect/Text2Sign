# service.py
import os, json, time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 내부 모듈(이미 보유)
import text2sign_retrieval_full as gen

# (선택) LLM 전처리: 키가 없어도 서버는 뜨도록 try-import
LLM_AVAILABLE = True
try:
    import llm_preprocessor as lp  # 당신이 쓰던 파일
except Exception:
    LLM_AVAILABLE = False

# ----------------- 환경 변수 -----------------
TEMPLATES_ROOT = os.getenv("TEMPLATES_ROOT", "data/templates_realsen_v3")
# 세미콜론(;)로 여러 경로를 연결할 수 있게 허용: a;b;c → 첫 번째만 사용(간단화)
if ";" in TEMPLATES_ROOT:
    # gen.build_template_index는 단일 root를 받도록 했으니 하나로 합칠 수 있게
    # 가장 앞 경로를 기본값으로 사용 (원하면 합친 폴더를 만들어도 됨)
    TEMPLATES_ROOT = TEMPLATES_ROOT.split(";")[0].strip()

REQUIRE_API_KEY = os.getenv("API_KEY")  # 선택: 헤더 검사
# LLM(OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_ON = bool(OPENROUTER_API_KEY and LLM_AVAILABLE)

# S3(선택)
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
USE_S3 = bool(S3_BUCKET)

s3 = None
if USE_S3:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)

# ----------------- FastAPI -----------------
app = FastAPI(title="Text2Sign API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- 모델 -----------------
class PlanToken(BaseModel):
    gloss: str
    seconds: float

class Plan(BaseModel):
    tokens: List[PlanToken]

class MakePlanRequest(BaseModel):
    text: str
    model: Optional[str] = "openrouter/auto"
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512

class SignTextRequest(BaseModel):
    text: str
    fps: int = 30
    fade_frames: int = 6
    gap_frames: int = 3
    pick: str = "median"

class SignPlanRequest(BaseModel):
    plan: Plan
    fps: int = 30
    fade_frames: int = 6
    gap_frames: int = 3
    pick: str = "median"

def _require_api_key(x_api_key: Optional[str]):
    if REQUIRE_API_KEY and x_api_key != REQUIRE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

def _sanitize_plan(raw_plan, allowed_glosses):
    """ llm_preprocessor.sanitize_plan 대체(없을 때 대비). """
    if LLM_AVAILABLE:
        try:
            return lp.sanitize_plan(raw_plan, allowed_glosses)
        except Exception:
            pass
    # 최소 보정: schema 강제
    tokens = raw_plan.get("tokens") if isinstance(raw_plan, dict) else None
    if not isinstance(tokens, list):
        raise HTTPException(500, "plan schema invalid")
    out = []
    for t in tokens:
        try:
            g = str(t["gloss"])
            s = float(t.get("seconds", 1.0))
            if g and s > 0:
                if allowed_glosses and (g not in allowed_glosses):
                    # 허용안되면 스킵
                    continue
                out.append({"gloss": g, "seconds": s})
        except Exception:
            continue
    if not out:
        raise HTTPException(500, "plan empty after sanitize")
    return {"tokens": out}

def _build_allowed_glosses(root):
    """템플릿 루트에서 글로스 목록 수집 (llm_preprocessor 없이도 동작)"""
    from pathlib import Path
    out = set()
    for p in Path(root).rglob("*.npz"):
        try:
            out.add(p.parent.name)
        except Exception:
            pass
    return sorted(out)

def _gen_jsonl_from_plan(plan: Plan, fps, fade, gap, pick) -> str:
    import tempfile, pathlib
    pathlib.Path("out").mkdir(parents=True, exist_ok=True)
    tmp_plan = pathlib.Path("out") / f"_plan_{int(time.time()*1000)}.json"
    tmp_out  = pathlib.Path("out") / f"_sign_{int(time.time()*1000)}.jsonl"
    tmp_plan.write_text(json.dumps(plan.dict(), ensure_ascii=False), encoding="utf-8")
    gen.generate_from_plan(str(tmp_plan), TEMPLATES_ROOT, str(tmp_out),
                           fps=fps, fade_frames=fade, gap_frames=gap, pick=pick)
    s = tmp_out.read_text(encoding="utf-8")
    # tmp_plan.unlink(missing_ok=True); tmp_out.unlink(missing_ok=True)
    return s

# ----------------- 엔드포인트 -----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "templates_root": TEMPLATES_ROOT,
        "llm_enabled": OPENROUTER_ON,
        "s3_enabled": USE_S3,
    }

@app.post("/plan", response_model=Plan)
def make_plan(req: MakePlanRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    if not OPENROUTER_ON:
        raise HTTPException(500, "LLM is not configured (OPENROUTER_API_KEY missing or llm_preprocessor not found).")
    allowed = _build_allowed_glosses(TEMPLATES_ROOT)
    raw_plan = lp.call_llm_make_plan(
        req.text, allowed,
        model=req.model, temperature=req.temperature, max_tokens=req.max_tokens
    )
    plan_dict = _sanitize_plan(raw_plan, set(allowed))
    # pydantic Plan으로 변환
    return Plan(tokens=[PlanToken(**t) for t in plan_dict["tokens"]])

@app.post("/sign/text")
def sign_from_text(req: SignTextRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    # 텍스트 → (lexicon 없이) 글로스 토큰화 + 리트리벌
    import pathlib
    pathlib.Path("out").mkdir(parents=True, exist_ok=True)
    tmp_out = pathlib.Path("out") / f"_sign_{int(time.time()*1000)}.jsonl"
    gen.generate(req.text, TEMPLATES_ROOT, str(tmp_out),
                 fps=req.fps, fade_frames=req.fade_frames, gap_frames=req.gap_frames,
                 pick=req.pick, lexicon_path=None)
    s = tmp_out.read_text(encoding="utf-8")
    return {"content_type": "application/x-ndjson", "size": len(s.encode("utf-8")), "jsonl": s}

@app.post("/sign/plan")
def sign_from_plan(req: SignPlanRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    jsonl = _gen_jsonl_from_plan(req.plan, req.fps, req.fade_frames, req.gap_frames, req.pick)
    return {"content_type": "application/x-ndjson", "size": len(jsonl.encode("utf-8")), "jsonl": jsonl}

@app.post("/sign/s3")
def sign_to_s3(req: SignPlanRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    if not USE_S3 or s3 is None or not S3_BUCKET:
        raise HTTPException(500, "S3 not configured")
    jsonl = _gen_jsonl_from_plan(req.plan, req.fps, req.fade_frames, req.gap_frames, req.pick)
    key = f"sign/{int(time.time())}/sign_sentence.jsonl"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=jsonl.encode("utf-8"),
        ContentType="application/x-ndjson",
        CacheControl="no-cache"
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=300
    )
    return {"bucket": S3_BUCKET, "key": key, "url": url}
