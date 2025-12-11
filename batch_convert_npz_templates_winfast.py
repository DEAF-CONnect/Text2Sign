# batch_convert_npz_templates_winfast.py
# ------------------------------------------------------------
# Windows-optimized converter
# AI-Hub OpenPose keypoints(*_keypoints.json) + morpheme(*_morpheme.json)
#   → unified npz (kpt[T,67,2], mask[T,67], fps)
#   → (optional) export per-gloss templates under templates_root/<gloss>/*.npz
#
# Focus on Windows perf:
#  - glob() instead of rglob() per video dir (fewer syscalls)
#  - single index build for keypoint roots (avoid repeated scans)
#  - orjson if available (bytes → loads)
#  - ProcessPoolExecutor with chunked tasks and minimal logging
#  - spawn-safe (Windows) multiprocessing
# ------------------------------------------------------------
from __future__ import annotations
import argparse, re, hashlib, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- fast JSON loader (orjson if available) ----------
try:
    import orjson as _oj
    def fast_load_bytes(b: bytes):
        return _oj.loads(b)
except Exception:
    import json as _json
    def fast_load_bytes(b: bytes):
        s = b.decode("utf-8", errors="ignore")
        return _json.loads(s)

# ---------- safe file → dict ----------
def safe_load_json(path: Path) -> dict:
    try:
        b = path.read_bytes()
        if not b or not b.strip():
            return {}
        try:
            return fast_load_bytes(b)
        except Exception:
            import json as _fallback
            return _fallback.loads(b.decode("utf-8", errors="ignore"))
    except Exception:
        return {}

# ---------- OpenPose → MediaPipe-like (upper25 + hands21+21) ----------
BODY_25 = 25
HAND_21 = 21
K = BODY_25 + HAND_21 + HAND_21  # 67

# minimal, dependency-free parser (keeps NaN for missings)

def _to_list(x):
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return []


def _take_person(jd: dict):
    ppl = jd.get("people")
    if isinstance(ppl, dict):
        return ppl
    if isinstance(ppl, list) and ppl:
        return ppl[0]
    return None


def _parse_xyc(flat: List[float], n: int):
    out = []
    m = len(flat)//3
    for i in range(n):
        if i < m:
            x = float(flat[3*i]); y = float(flat[3*i+1]); c = float(flat[3*i+2])
        else:
            x = float("nan"); y = float("nan"); c = 0.0
        out.append({"x": x, "y": y, "visibility": c})
    return out


def _pad_or_trim(pts: List[dict], n: int):
    if len(pts) == n:
        return pts
    if len(pts) > n:
        return pts[:n]
    return pts + ([{"x": float("nan"), "y": float("nan"), "visibility": 0.0}]*(n-len(pts)))


def convert_frame_openpose_to_mp(jd: dict, width: int, height: int, normalize: bool) -> dict:
    p = _take_person(jd)
    if p is None:
        nan_pose = [{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * BODY_25
        nan_hand = [{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * HAND_21
        return {"pose": nan_pose, "left_hand": nan_hand, "right_hand": nan_hand}
    pose_flat = _to_list(p.get("pose_keypoints_2d", []))
    lh_flat   = _to_list(p.get("hand_left_keypoints_2d", []))
    rh_flat   = _to_list(p.get("hand_right_keypoints_2d", []))
    pose = _pad_or_trim(_parse_xyc(pose_flat, BODY_25), BODY_25)
    lhd  = _pad_or_trim(_parse_xyc(lh_flat,   HAND_21), HAND_21)
    rhd  = _pad_or_trim(_parse_xyc(rh_flat,   HAND_21), HAND_21)
    if normalize:
        w = max(1, int(width)); h = max(1, int(height))
        for arr in (pose, lhd, rhd):
            for q in arr:
                x = q.get("x"); y = q.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if np.isfinite(x) and np.isfinite(y):
                        q["x"] = float(x)/w
                        q["y"] = float(y)/h
    return {"pose": pose, "left_hand": lhd, "right_hand": rhd}

# ---------- pack to fixed arrays ----------

def pack_fixed_k(mp_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((K,2), np.float32)
    msk = np.zeros((K,),  np.uint8)
    def fill(block, off, lim):
        block = block or []
        n = min(len(block), lim)
        for i in range(n):
            d = block[i] or {}
            x = d.get("x", np.nan); y = d.get("y", np.nan)
            ok = np.isfinite(x) and np.isfinite(y) and (0.0 <= float(x) <= 1.0) and (0.0 <= float(y) <= 1.0)
            pts[off+i,0] = float(x) if ok else 0.0
            pts[off+i,1] = float(y) if ok else 0.0
            msk[off+i]   = 1 if ok else 0
    fill(mp_dict.get("pose"), 0, BODY_25)
    fill(mp_dict.get("left_hand"), BODY_25, HAND_21)
    fill(mp_dict.get("right_hand"), BODY_25+HAND_21, HAND_21)
    return pts, msk

# ---------- helpers ----------

def save_npz(path: Path, arrays: Dict[str, np.ndarray], compressed: bool=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def round_frame(t: float, fps: int) -> int:
    return int(round(float(t) * fps))

# time unit auto-detect (s vs ms) for morpheme JSON

def load_morpheme_segments(morpheme_path: Path, fps: int) -> List[Tuple[int,int,str]]:
    d = safe_load_json(morpheme_path)
    rows = d.get("data", []) if isinstance(d, dict) else []
    max_end = 0.0
    for r in rows:
        try:
            max_end = max(max_end, float(r.get("end", 0.0)))
        except Exception:
            pass
    scale = 1000.0 if max_end > 1000.0 else 1.0
    segs: List[Tuple[int,int,str]] = []
    for seg in rows:
        attrs = seg.get("attributes") or []
        if attrs and isinstance(attrs, list) and isinstance(attrs[0], dict):
            gloss = attrs[0].get("name", "UNK")
        else:
            gloss = seg.get("gloss", "UNK")
        try:
            s = round_frame(float(seg.get("start", 0.0))/scale, fps)
            e = round_frame(float(seg.get("end",   0.0))/scale, fps)
        except Exception:
            s, e = 0, 0
        if e > s:
            segs.append((max(0,s), max(0,e), str(gloss)))
    return segs

# shard path helper (fs/sen buckets)

FS_SEN_RE = re.compile(r"_(FS|SEN)(\d+)_")

def compute_shard(base: str, mode: str = "fs_sen_bucket", bucket_size: int = 1000) -> str:
    if mode == "none":
        return ""
    m = FS_SEN_RE.search(base)
    if mode == "fs_sen":
        return (m.group(1) if m else "OTHER")
    if mode == "fs_sen_bucket":
        if not m:
            return "OTHER"
        kind, num = m.group(1), int(m.group(2))
        start = ((num - 1)//bucket_size)*bucket_size + 1
        end = start + bucket_size - 1
        return f"{kind}/{start:04d}-{end:04d}"
    if mode == "hash2":
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return f"{h[:2]}/{h[2:4]}"
    return "OTHER"

# build key index once (choose parent folder that has most frames)

BASE_RE = re.compile(r"^(?P<base>.+?)_\d{6,}_keypoints\.json$")

def build_key_index(key_root: Path) -> Dict[str, Path]:
    counts: Dict[Tuple[str, Path], int] = {}
    # Windows: avoid deep rglob; but we need one-time global scan
    for jf in key_root.rglob("*_keypoints.json"):
        m = BASE_RE.match(jf.name)
        if not m:
            continue
        base = m.group("base")
        parent = jf.parent
        key = (base, parent)
        counts[key] = counts.get(key, 0) + 1
    best: Dict[str, Tuple[int, Path]] = {}
    for (base, parent), cnt in counts.items():
        cur = best.get(base)
        if cur is None or cnt > cur[0]:
            best[base] = (cnt, parent)
    index: Dict[str, Path] = {base: parent for base, (cnt, parent) in best.items()}
    print(f"[index] bases={len(index)}")
    return index

# convert one video folder → npz (and optional templates)

def convert_one(base: str, morpheme_path: Path, key_dir: Path, a: dict):
    fps = a["fps"]; width=a["width"]; height=a["height"]
    assume_norm = a["assume_normalized"]
    save_mode=a["save_mode"]
    
    shard_rel = compute_shard(base, a["shard_mode"], a["bucket_size"])
    out_npz = Path(a["out_root"]) / shard_rel / f"{base}.npz"
    tpl_dir = Path(a["templates_root"]) / shard_rel / base if a["export_templates"] else None

    # skip decisions
    legacy_npz = Path(a["out_root"]) / f"{base}.npz"
    legacy_tpl_dir = Path(a["templates_root"]) / base if a["export_templates"] else None

    npz_exists = out_npz.exists() or legacy_npz.exists()
    tpl_exists = False
    if a["export_templates"]:
        d1 = tpl_dir if tpl_dir else Path(".")
        d2 = legacy_tpl_dir if legacy_tpl_dir else Path(".")
        tpl_exists = (d1.exists() and any(d1.rglob("*.npz"))) or (d2.exists() and any(d2.rglob("*.npz")))

    if a["skip_existing_npz"] and npz_exists and (not a["export_templates"] or (a["skip_existing_templates"] and tpl_exists)):
        return base, "skipped_all"

    # list frames fast (no deep recursion)
    if not key_dir or not key_dir.exists():
        return base, "no_key_dir"
    keyfiles = [p for p in key_dir.glob("*_keypoints.json") if p.is_file() and p.stat().st_size > 0]
    keyfiles.sort()
    if not keyfiles:
        return base, "empty_key_dir"

    # segments
    segs = load_morpheme_segments(morpheme_path, fps=fps)

    frames_pts = []
    frames_msk = []
    for kf in keyfiles:
        jd = safe_load_json(kf)
        ppl = jd.get("people") if isinstance(jd, dict) else None
        if not ppl or (isinstance(ppl, list) and len(ppl) == 0):
            # zero pad frame
            frames_pts.append(np.zeros((K,2), np.float32))
            frames_msk.append(np.zeros((K,), np.uint8))
            continue
        try:
            mp = convert_frame_openpose_to_mp(jd, width, height, normalize=not assume_norm)
            pts, msk = pack_fixed_k(mp)
        except Exception:
            pts = np.zeros((K,2), np.float32)
            msk = np.zeros((K,), np.uint8)
        frames_pts.append(pts)
        frames_msk.append(msk)

    kpt = np.stack(frames_pts, axis=0)
    msk = np.stack(frames_msk, axis=0)
    save_npz(out_npz, {"kpt": kpt.astype(np.float32), "mask": msk.astype(np.uint8), "fps": np.array(int(fps))}, compressed=(save_mode=="compressed"))

    if not a["export_templates"]:
        return base, "ok_npz"

    # export templates
    if a["skip_existing_templates"] and tpl_exists:
        return base, "npz_ok_tpl_skipped"

    export_templates(kpt, msk, segs, tpl_dir, a["min_len"], a["max_len"], a["long_policy"], a["chunk_stride"], save_mode)
    return base, "ok_exported"


def export_templates(kpt: np.ndarray, mask: np.ndarray, segs: List[Tuple[int,int,str]], out_root: Path,
                     min_len: int, max_len: int, long_policy: str, chunk_stride: Optional[int], save_mode: str):
    out_root.mkdir(parents=True, exist_ok=True)
    counters: Dict[str,int] = {}
    T = kpt.shape[0]
    if chunk_stride is None:
        chunk_stride = max(1, max_len//2)
    total = 0
    for s, e, gloss in segs:
        s = max(0, min(T, s)); e = max(0, min(T, e))
        L = e - s
        if L <= 0 or L < min_len:
            continue
        parts: List[Tuple[int,int]]
        if L > max_len:
            if long_policy == "skip":
                continue
            elif long_policy == "center":
                mid = (s+e)//2
                s2 = max(0, mid - max_len//2)
                e2 = min(T, s2 + max_len)
                parts = [(s2, e2)]
            elif long_policy == "chunk":
                parts = []
                pos = s
                while pos + min_len <= e:
                    s2 = pos; e2 = min(e, s2 + max_len)
                    if e2 - s2 >= min_len:
                        parts.append((s2, e2))
                    pos += chunk_stride
            else:
                continue
        else:
            parts = [(s, e)]
        for s2, e2 in parts:
            clip_k = kpt[s2:e2]
            clip_m = mask[s2:e2]
            gdir = out_root / gloss
            gdir.mkdir(parents=True, exist_ok=True)
            idx = counters.get(gloss, 0)
            outp = gdir / f"sample_{idx:03d}.npz"
            save_npz(outp, {"kpt": clip_k.astype(np.float32), "mask": clip_m.astype(np.uint8)}, compressed=(save_mode=="compressed"))
            counters[gloss] = idx + 1
            total += 1
    print(f"[templates] exported={total} → {out_root}")


# ---------- main (Windows-safe multiprocessing) ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morpheme_root", required=True)
    ap.add_argument("--keypoint_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--assume_normalized", action="store_true")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--limit", type=int, default=0)

    # storage/speed
    ap.add_argument("--save_mode", choices=["uncompressed","compressed"], default="uncompressed")
    ap.add_argument("--workers", type=int, default=1)

    # templates
    ap.add_argument("--export_templates", action="store_true")
    ap.add_argument("--templates_root", type=str, default="data/templates_realsen_v3")
    ap.add_argument("--min_len", type=int, default=18)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--long_policy", choices=["skip","center","chunk"], default="center")
    ap.add_argument("--chunk_stride", type=int, default=None)

    # skip
    ap.add_argument("--skip_existing_npz", action="store_true")
    ap.add_argument("--skip_existing_templates", action="store_true")

    # sharding
    ap.add_argument("--shard_mode", choices=["none","fs_sen","fs_sen_bucket","hash2"], default="fs_sen_bucket")
    ap.add_argument("--bucket_size", type=int, default=1000)

    args = ap.parse_args()

    mor_root = Path(args.morpheme_root)
    key_root = Path(args.keypoint_root)
    out_root = Path(args.out_root)

    mor_files = sorted(mor_root.rglob("*_morpheme.json"))
    print(f"[scan] morpheme files = {len(mor_files)}")
    if args.limit and args.limit > 0:
        mor_files = mor_files[:args.limit]

    key_index = build_key_index(key_root)

    a = {
        "fps": args.fps,
        "assume_normalized": args.assume_normalized,
        "width": args.width,
        "height": args.height,
        "save_mode": args.save_mode,
        "export_templates": args.export_templates,
        "templates_root": args.templates_root,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "long_policy": args.long_policy,
        "chunk_stride": args.chunk_stride,
        "skip_existing_npz": args.skip_existing_npz,
        "skip_existing_templates": args.skip_existing_templates,
        "out_root": str(out_root),
        "shard_mode": args.shard_mode,
        "bucket_size": args.bucket_size,
    }

    tasks: List[Tuple[str, Path]] = []
    for mor in mor_files:
        base = mor.stem.replace("_morpheme", "")
        kdir = key_index.get(base)
        tasks.append((base, mor, kdir))

    total = len(tasks)
    print(f"[tasks] total={total}")
    if total == 0:
        print("[done] nothing to do.")
        return

    if args.workers <= 1:
        done = 0
        for base, mor, kdir in tasks:
            status = ""
            try:
                base, status = convert_one(base, mor, kdir, a)
            except Exception as e:
                status = f"worker_exception:{e}"
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[{done}/{total}] {base}: {status}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Windows: protect entry point in if __name__ == '__main__'
        # We are in main(), so safe.
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(convert_one, base, mor, kdir, a) for base, mor, kdir in tasks]
            for fut in as_completed(futs):
                try:
                    base, status = fut.result()
                except Exception as e:
                    base, status = "unknown", f"worker_exception:{e}"
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[{done}/{total}] {base}: {status}")

    print("[done] conversion complete.")


if __name__ == "__main__":
    # On Windows, spawn is default; make sure main guard is present (it is).
    main()
