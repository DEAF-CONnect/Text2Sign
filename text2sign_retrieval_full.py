import os, json, argparse, re, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- 유틸 ----------
def simple_edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return dp[-1]

def sanitize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[..., 0] = np.clip(arr[..., 0], 0.0, 1.0)
    arr[..., 1] = np.clip(arr[..., 1], 0.0, 1.0)
    return arr

def resample_seq(arr: np.ndarray, out_len: int) -> np.ndarray:
    T, K = arr.shape[0], arr.shape[1]
    if out_len <= 1: out_len = 2
    if T == out_len: return arr.copy()
    x  = np.arange(T)
    xi = np.linspace(0, T-1, out_len)
    out = np.zeros((out_len, K, 2), dtype=np.float32)
    for k in range(K):
        for c in range(2):
            out[:, k, c] = np.interp(xi, x, arr[:, k, c])
    return out

def crossfade(a: np.ndarray, b: np.ndarray, fade: int) -> np.ndarray:
    if fade <= 0: return np.concatenate([a, b], axis=0)
    F = min(fade, a.shape[0], b.shape[0])
    if F == 0: return np.concatenate([a, b], axis=0)
    alpha = np.linspace(0, 1, F, endpoint=True).reshape(F, 1, 1).astype(np.float32)
    head = a[:-F] if a.shape[0] > F else np.zeros((0, a.shape[1], 2), dtype=np.float32)
    mix  = (1-alpha)*a[-F:] + alpha*b[:F]
    tail = b[F:] if b.shape[0] > F else np.zeros((0, b.shape[1], 2), dtype=np.float32)
    return np.concatenate([head, mix, tail], axis=0)

def hold_last(arr: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0: return np.zeros((0, arr.shape[1], 2), dtype=np.float32)
    return np.repeat(arr[-1:], frames, axis=0)

# ---------- 경로/manifest 헬퍼 ----------
def split_roots(templates_root: str | List[str]) -> List[str]:
    if isinstance(templates_root, (list, tuple)):
        return [str(x) for x in templates_root]
    return [t.strip() for t in str(templates_root).split(",") if t.strip()]

def load_json_file(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding='utf-8'))

def try_load_manifest(manifest_path: Optional[str], roots: List[str]) -> Dict[str, dict]:
    """
    1) --manifest 가 주어지면 그것을 로드
    2) 없으면 각 root 하위의 manifest.json을 찾아서 병합(뒤에 오는 것이 우선)
    """
    merged: Dict[str, dict] = {}
    if manifest_path:
        m = load_json_file(manifest_path)
        for k, v in m.items():
            if isinstance(v, dict) and 'source' not in v:
                v = {**v, 'source': v.get('source', 'unknown')}
            merged[k] = v
        return merged

    for r in roots:
        mp = Path(r) / 'manifest.json'
        if mp.exists():
            m = load_json_file(str(mp))
            for k, v in m.items():
                if isinstance(v, dict):
                    v = {**v, 'source': v.get('source', Path(r).name), 'root': v.get('root', Path(r).name)}
                merged[k] = v
    return merged

def resolve_template_path(entry: dict, roots: List[str]) -> Optional[str]:
    """
    entry['path'] 가 상대경로면 roots와 힌트(root|source)로 실제 경로 생성.
    s3:// 또는 / 또는 ./ 로 시작하면 그대로 반환.
    """
    path = entry.get('path')
    if not path:
        return None
    if str(path).startswith(('s3://','/','./')):
        return str(path)

    hint = entry.get('root') or entry.get('source')
    if hint:
        for r in roots:
            if Path(r).name == hint:
                return str(Path(r) / path)

    # 힌트가 없으면 첫 번째 루트 기준
    return str(Path(roots[0]) / path)

# ---------- 템플릿 인덱스 ----------
def iter_npz_paths(roots: List[str]):
    for root in roots:
        for p in sorted(Path(root).rglob("*.npz")):
            yield p

def build_template_index(templates_root: str | List[str], min_len: int=4, max_len: int=600):
    roots = split_roots(templates_root)
    idx: Dict[str, List[Path]] = {}
    for p in iter_npz_paths(roots):
        try:
            d = np.load(p)
            if "kpt" not in d: continue
            T = int(d["kpt"].shape[0])
            if not (min_len <= T <= max_len): continue
            gloss = p.parent.name
            idx.setdefault(gloss, []).append(p)
        except Exception:
            continue
    return idx  # {gloss: [npz paths]}

def pick_template(cands: List[Path], strategy="median") -> Path:
    if strategy == "random": return random.choice(cands)
    lens = []
    for p in cands:
        d = np.load(p); lens.append(d["kpt"].shape[0])
    order = np.argsort(lens)
    return cands[int(order[len(order)//2])]

# ---------- 토크나이즈 & 사전 매핑 ----------
_PUNCT = r"[,\.\!\?\:\;\(\)\[\]\{\}…~\-_/]"
def tokenize(text: str) -> List[str]:
    text = re.sub(f"({_PUNCT})", r" \1 ", text)
    toks = [t.strip() for t in text.split() if t.strip()]
    return toks

def load_lexicon(path: Optional[str]) -> Dict[str, Dict]:
    if not path: return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d

def map_tokens_to_glosses(tokens: List[str], lexicon: Dict[str, Dict]) -> List[Tuple[str, float]]:
    out = []
    for t in tokens:
        info = lexicon.get(t)
        if info:
            if "pause" in info:
                out.append((f"<PAUSE_{t}>", float(info["pause"])))
            else:
                secs = float(info.get("seconds", 1.0))
                gloss_seq = info["gloss"] if isinstance(info["gloss"], list) else [info["gloss"]]
                for g in gloss_seq:
                    out.append((g, secs))
        else:
            if re.match(_PUNCT, t):
                out.append((f"<PAUSE_{t}>", 0.3))
            else:
                out.append((t, 1.0))
    return out

# ---------- OOV 대체 ----------
def find_best_gloss(name: str, index: Dict[str, List[Path]], max_ed=2) -> Optional[str]:
    if name in index: return name
    cands = [g for g in index.keys() if name in g or g in name]
    if cands: return min(cands, key=lambda g: abs(len(g)-len(name)))
    best, best_ed = None, 10**9
    for g in index.keys():
        ed = simple_edit_distance(name, g)
        if ed < best_ed:
            best_ed, best = ed, g
    return best if best_ed <= max_ed else None

# ---------- 출력 (JSONL / TXT / JSON) ----------
def _ensure_parent_dir(out_path: str):
    dir_path = os.path.dirname(out_path)
    if dir_path:  # 현재 폴더 저장("mapped.txt")인 경우 "" 방지
        os.makedirs(dir_path, exist_ok=True)

def to_jsonl(arr: np.ndarray, out_path: str, fps: int, token_track: List[str]):
    pose_n, lh_n, rh_n = 25, 21, 21
    assert arr.shape[1] == (pose_n+lh_n+rh_n)
    lines = []
    for i in range(arr.shape[0]):
        ms = int((i/fps)*1000)
        pose = [{"x":float(x), "y":float(y), "visibility":1.0} for (x,y) in arr[i,:pose_n]]
        lhd  = [{"x":float(x), "y":float(y), "visibility":1.0} for (x,y) in arr[i,pose_n:pose_n+lh_n]]
        rhd  = [{"x":float(x), "y":float(y), "visibility":1.0} for (x,y) in arr[i,pose_n+lh_n:]]
        rec = {"t":ms, "pose":pose, "left_hand":lhd, "right_hand":rhd, "token":token_track[i]}
        lines.append(json.dumps(rec, ensure_ascii=False))
    _ensure_parent_dir(out_path)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ wrote {len(lines)} frames → {out_path}")

def to_txt(arr: np.ndarray, out_path: str, fps: int, token_track: List[str]):
    """사람이 읽기 좋은 간단 TXT: 00012 | token=안녕 | x=0.523, y=0.412"""
    lines = []
    for i, token in enumerate(token_track):
        x_mean = float(np.mean(arr[i, :, 0]))
        y_mean = float(np.mean(arr[i, :, 1]))
        lines.append(f"{i:05d} | token={token} | x={x_mean:.3f}, y={y_mean:.3f}")
    _ensure_parent_dir(out_path)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ wrote {len(lines)} frames (TXT mode) → {out_path}")

def to_json(arr: np.ndarray, out_path: str, fps: int, token_track: List[str]):
    """전체 리스트형 JSON 출력 (시작과 끝에 [ ... ]로 감싼 완전한 JSON 구조)"""
    pose_n, lh_n, rh_n = 25, 21, 21
    assert arr.shape[1] == (pose_n + lh_n + rh_n)
    
    # 모든 프레임을 리스트 형태로 구성
    data = []
    for i in range(arr.shape[0]):
        ms = int((i / fps) * 1000)
        pose = [{"x": float(x), "y": float(y), "visibility": 1.0} for (x, y) in arr[i, :pose_n]]
        lhd = [{"x": float(x), "y": float(y), "visibility": 1.0} for (x, y) in arr[i, pose_n:pose_n + lh_n]]
        rhd = [{"x": float(x), "y": float(y), "visibility": 1.0} for (x, y) in arr[i, pose_n + lh_n:]]
        data.append({
            "t": ms,
            "pose": pose,
            "left_hand": lhd,
            "right_hand": rhd,
            "token": token_track[i]
        })
    
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 👇 핵심: JSON 배열 전체를 저장 (시작은 [, 끝은 ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, frame in enumerate(data):
            json.dump(frame, f, ensure_ascii=False, indent=2)
            if i != len(data) - 1:
                f.write(",\n")
        f.write("\n]")
    
    print(f"✅ wrote {len(data)} frames (JSON array mode) → {out_path}")


# ---------- 일반 텍스트 기반 생성 ----------
def generate(text: str, templates_root: str | List[str], out_path: str,
             fps=30, fade_frames=6, gap_frames=3, default_seconds=1.0,
             pick="median", lexicon_path: Optional[str]=None,
             manifest: Optional[Dict[str, dict]]=None):
    roots = split_roots(templates_root)
    index = build_template_index(roots)
    if not index and not manifest:
        raise RuntimeError(f"No templates found under {roots}")
    lexicon = load_lexicon(lexicon_path)
    raw_tokens = tokenize(text)
    seq_plan = map_tokens_to_glosses(raw_tokens, lexicon)
    _generate_seq(seq_plan, index, out_path, fps, fade_frames, gap_frames, pick, roots, manifest)

# ---------- plan.json 기반 생성 ----------
def generate_from_plan(plan_json: str, templates_root: str | List[str], out_path: str,
                       fps=30, fade_frames=6, gap_frames=3, pick="median",
                       manifest: Optional[Dict[str, dict]]=None):
    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    # 기대 형식: {"tokens":[{"gloss":..., "seconds":...}, ...]} 또는 {"processed":[...]} 등
    if "tokens" in plan:
        seq_plan = [(t["gloss"], float(t.get("seconds", 1.0))) for t in plan["tokens"]]
    elif "processed" in plan and isinstance(plan["processed"], list):
        seq_plan = [(str(g), 1.0) for g in plan["processed"]]
    else:
        raise RuntimeError("plan_json 형식을 이해할 수 없습니다. 'tokens' 또는 'processed' 키가 필요합니다.")

    roots = split_roots(templates_root)
    index = build_template_index(roots)
    _generate_seq(seq_plan, index, out_path, fps, fade_frames, gap_frames, pick, roots, manifest)

# ---------- 공통 내부 로직 ----------
def _load_npz_from_manifest(gloss: str, roots: List[str], manifest: Dict[str, dict]) -> Optional[np.ndarray]:
    entry = (manifest.get(gloss) or manifest.get(gloss.upper()) or manifest.get(gloss.lower()))
    if not entry: return None
    full = resolve_template_path(entry, roots)
    if not full: return None
    if full.startswith("s3://"):
        # 여기서는 s3 직접 로드는 생략(사전 다운로드 권장). 필요하면 boto3로 확장 가능.
        return None
    p = Path(full)
    if not p.exists(): return None
    d = np.load(p)
    if "kpt" not in d: return None
    return sanitize(d["kpt"].astype(np.float32))

def _generate_seq(seq_plan, index, out_path, fps, fade_frames, gap_frames, pick, roots, manifest):
    assembled = []
    token_track = []
    for item, secs in seq_plan:
        if item.startswith("<PAUSE_") or item == "<PAUSE>":
            if assembled:
                hold = hold_last(assembled[-1], int(round(secs*fps)))
                assembled.append(hold)
                token_track.extend([item]*hold.shape[0])
            continue

        seg = None

        # 1) manifest 우선
        if manifest:
            seg = _load_npz_from_manifest(item, roots, manifest)

        # 2) 폴백: 기존 retrieval 인덱스
        if seg is None:
            gloss = find_best_gloss(item, index)
            if gloss is None:
                print(f"[WARN] no template for '{item}' → idle {secs}s")
                if assembled:
                    hold = hold_last(assembled[-1], int(round(secs*fps)))
                    assembled.append(hold)
                    token_track.extend([item]*hold.shape[0])
                continue
            npz = pick_template(index[gloss], strategy=pick)
            d = np.load(npz)
            if "kpt" not in d:
                print(f"[WARN] template '{gloss}' has no kpt → idle {secs}s")
                if assembled:
                    hold = hold_last(assembled[-1], int(round(secs*fps)))
                    assembled.append(hold)
                    token_track.extend([item]*hold.shape[0])
                continue
            seg = sanitize(d["kpt"].astype(np.float32))

        # 리샘플링
        seg = resample_seq(seg, max(2, int(round(secs*fps))))

        # 조립
        if not assembled:
            assembled.append(seg)
            token_track.extend([item]*seg.shape[0])
        else:
            merged = crossfade(assembled[-1], seg, fade_frames)
            assembled[-1] = merged
            F = min(fade_frames, seg.shape[0])
            if F > 0:
                token_track[-F:] = [f"{token_track[-1]}+{item}"]*F
            if gap_frames > 0:
                gap = hold_last(merged, gap_frames)
                assembled.append(gap)
                token_track.extend([item]*gap.shape[0])
            assembled.append(seg)
            token_track.extend([item]*seg.shape[0])

    chunks = [a for a in assembled if a.size>0]
    if not chunks: raise RuntimeError("No frames generated")
    out = sanitize(np.concatenate(chunks, axis=0))
    if len(token_track) < out.shape[0]:
        token_track += [token_track[-1]]*(out.shape[0]-len(token_track))
    elif len(token_track) > out.shape[0]:
        token_track = token_track[:out.shape[0]]

    # 확장자에 따라 자동 분기
    low = str(out_path).lower()
    if   low.endswith(".txt"):
        to_txt(out, out_path, fps=fps, token_track=token_track)
    elif low.endswith(".json"):
        to_json(out, out_path, fps=fps, token_track=token_track)
    else:
        to_jsonl(out, out_path, fps=fps, token_track=token_track)

    print("🎬 Done.")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_root", required=True, help="여러 경로는 콤마로 구분 (예: /a/b,/c/d)")
    ap.add_argument("--text", help="텍스트 입력 (plan_json 없을 때)")
    ap.add_argument("--plan_json", help="LLM 전처리 결과 JSON 파일 (tokens:[{gloss,seconds}] 또는 processed:[])")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade_frames", type=int, default=6)
    ap.add_argument("--gap_frames", type=int, default=3)
    ap.add_argument("--default_seconds", type=float, default=1.0)
    ap.add_argument("--pick", choices=["median","random"], default="median")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--manifest", default=None, help="병합 manifest 경로(선택). 미지정 시 각 root/manifest.json 자동 탐색")
    args = ap.parse_args()

    roots = split_roots(args.templates_root)
    manifest = try_load_manifest(args.manifest, roots)

    if args.plan_json:
        generate_from_plan(args.plan_json, roots, args.out,
                           fps=args.fps, fade_frames=args.fade_frames, gap_frames=args.gap_frames,
                           pick=args.pick, manifest=manifest)
    else:
        if not args.text:
            raise RuntimeError("Either --text or --plan_json is required.")
        generate(args.text, roots, args.out,
                 fps=args.fps, fade_frames=args.fade_frames, gap_frames=args.gap_frames,
                 default_seconds=args.default_seconds, pick=args.pick, lexicon_path=args.lexicon,
                 manifest=manifest)

if __name__ == "__main__":
    main()
