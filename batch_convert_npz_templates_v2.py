#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Hub(OpenPose JSON + morpheme) → MediaPipe-like npz (kpt[T,67,2], mask[T,67], fps)
옵션으로 글로스 템플릿 추출까지.

개선 사항(v2):
- keypoint 루트 여러 개 지원: --keypoint_roots "pathA;pathB"
- 파일 패턴 확장: *_keypoints.json / *_keypoint.json / *_pose_keypoints.json / *_keypoints2d.json
- BASE_RE 완화: 프레임 4자리 이상 + 다양한 접미사 수용
- 인덱스 실패 시 폴백 검색(각 루트에서 base별 파일 직접 탐색)
- orjson(있으면) 사용, 빈/깨진 JSON 안전 로딩, 제로 패딩 유지
- 샤딩(fs/sen bucket 등), 스킵 옵션, 멀티프로세스

사용 예:
python batch_convert_npz_templates_v2.py \
  --morpheme_root data/01_crowd_morpheme \
  --keypoint_roots "data/01_crowd_keypoint;data/02_crowd_keypoint" \
  --out_root data/npz_crowd_v2 \
  --export_templates \
  --templates_root data/templates_crowd_v2 \
  --fps 30 --workers 8 --save_mode uncompressed \
  --skip_existing_npz --skip_existing_templates \
  --shard_mode fs_sen_bucket --bucket_size 1000 \
  --min_len 18 --max_len 120 --long_policy center
"""

import argparse, re, hashlib, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- JSON loader (fast + safe) ----------
try:
    import orjson as _oj
    def _fast_json_load_bytes(b: bytes):
        return _oj.loads(b)
except Exception:
    def _fast_json_load_bytes(b: bytes):
        s = b.decode("utf-8", errors="ignore")
        return json.loads(s)

def safe_load_json(path: str):
    """빈/깨진 JSON도 안전하게 dict 반환(실패 시 {})."""
    p = Path(path)
    try:
        b = p.read_bytes()
        if not b or b.strip() == b"":
            return {}
        try:
            return _fast_json_load_bytes(b)
        except Exception:
            s = b.decode("utf-8", errors="ignore").strip()
            if not s:
                return {}
            return json.loads(s)
    except Exception:
        return {}

# ---------- 사용자 제공: OpenPose→MediaPipe 변환 ----------
# 프로젝트에 맞는 변환기를 준비하세요.
from aihub2mediapipe import convert_frame_openpose_to_mp

# ---------- skeleton sizes ----------
UPPER_CNT, LEFT_HAND_CNT, RIGHT_HAND_CNT = 25, 21, 21
K = UPPER_CNT + LEFT_HAND_CNT + RIGHT_HAND_CNT  # 67

# ---------- save helpers ----------
def save_npz(path: str, arrays: Dict[str, np.ndarray], mode: str = "uncompressed"):
    if mode == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)

# ---------- utils ----------
def round_frame(t: float, fps: int) -> int:
    return int(round(float(t) * fps))

def normalize_pixels_inplace(mp_dict: Dict, width: int, height: int):
    w, h = max(1, int(width)), max(1, int(height))
    for part in ("pose", "left_hand", "right_hand"):
        for p in (mp_dict.get(part) or []):
            if "x" in p and "y" in p:
                p["x"] = float(p["x"]) / w
                p["y"] = float(p["y"]) / h

def pack_fixed_k(mp_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((K, 2), dtype=np.float32)
    msk = np.zeros((K,), dtype=np.uint8)
    def fill(block, offset, limit):
        block = block or []
        n = min(len(block), limit)
        for i in range(n):
            p = block[i] or {}
            x, y = p.get("x", np.nan), p.get("y", np.nan)
            ok = np.isfinite(x) and np.isfinite(y) and (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
            pts[offset+i, 0] = float(x) if ok else 0.0
            pts[offset+i, 1] = float(y) if ok else 0.0
            msk[offset+i]    = 1 if ok else 0
    fill(mp_dict.get("pose"), 0, UPPER_CNT)
    fill(mp_dict.get("left_hand"), UPPER_CNT, LEFT_HAND_CNT)
    fill(mp_dict.get("right_hand"), UPPER_CNT + LEFT_HAND_CNT, RIGHT_HAND_CNT)
    return pts, msk

# ---------- shard path ----------
def compute_shard(base: str, mode: str = "fs_sen_bucket", bucket_size: int = 1000) -> str:
    """
    base 예: 'NIA_SL_FS0003_CROWD01_F'
    반환:
      - none: ''
      - fs_sen: 'FS' or 'SEN'
      - fs_sen_bucket: 'FS/0001-1000'
      - hash2: 'ab/cd'
    """
    if mode == "none":
        return ""
    m = re.search(r'_(FS|SEN)(\d+)_', base)
    if mode == "fs_sen":
        return (m.group(1) if m else "OTHER")
    if mode == "fs_sen_bucket":
        if not m: return "OTHER"
        kind, num = m.group(1), int(m.group(2))
        start = ((num - 1) // bucket_size) * bucket_size + 1
        end = start + bucket_size - 1
        return f"{kind}/{start:04d}-{end:04d}"
    if mode == "hash2":
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return f"{h[:2]}/{h[2:4]}"
    return "OTHER"

# ---------- key folder index (multi roots) ----------
# 파일명 예: NIA_SL_FS0992_CROWD17_F_000001_keypoints.json
# 4자리 이상 프레임 + 다양한 접미사 허용
BASE_RE = re.compile(
    r"^(?P<base>.+?)_\d{4,}_(?:keypoints?|pose_keypoints|keypoints2d)\.json$"
)
KEY_PATTERNS = [
    "*_keypoints.json",
    "*_keypoint.json",
    "*_pose_keypoints.json",
    "*_keypoints2d.json",
]

def build_key_index_one_root(key_root: Path) -> Dict[str, Path]:
    """한 루트에서 base별로 프레임이 가장 많이 모인 폴더를 선택."""
    counts: Dict[Tuple[str, Path], int] = {}
    for pat in KEY_PATTERNS:
        for jf in key_root.rglob(pat):
            m = BASE_RE.match(jf.name)
            if not m:
                continue
            base = m.group("base")
            parent = jf.parent
            counts[(base, parent)] = counts.get((base, parent), 0) + 1
    best: Dict[str, Tuple[int, Path]] = {}
    for (base, parent), cnt in counts.items():
        cur = best.get(base)
        if cur is None or cnt > cur[0]:
            best[base] = (cnt, parent)
    index: Dict[str, Path] = {base: parent for base, (cnt, parent) in best.items()}
    return index

def build_key_index_multi(roots: List[str]) -> Dict[str, Path]:
    merged: Dict[str, Tuple[int, Path]] = {}
    for r in roots:
        sub = build_key_index_one_root(Path(r))
        for base, parent in sub.items():
            # rough count for prioritization
            cnt = sum(1 for _ in parent.glob("*.json"))
            if base not in merged or cnt > merged[base][0]:
                merged[base] = (cnt, parent)
    index: Dict[str, Path] = {b: p for b, (c, p) in merged.items()}
    print(f"📚 key-index built across {len(roots)} roots: {len(index)} bases")
    return index

def fallback_find_key_dir(roots: List[str], base: str) -> Optional[Path]:
    """인덱스에 없을 때, 직접 파일을 찾아서 그 부모 폴더를 반환."""
    for r in roots:
        kroot = Path(r)
        for suf in ["keypoints.json","keypoint.json","pose_keypoints.json","keypoints2d.json"]:
            cand = next(iter(kroot.rglob(f"{base}_*_{suf}")), None)
            if cand is not None:
                return cand.parent
    return None

# ---------- morpheme segments (auto unit detect) ----------
def load_morpheme_segments(morpheme_path: str, fps: int, debug: bool=False) -> List[Tuple[int,int,str]]:
    data = safe_load_json(morpheme_path)
    rows = data.get("data", [])
    max_end = max((float(r.get("end", 0.0)) for r in rows), default=0.0)
    time_scale = 1000.0 if max_end > 1000.0 else 1.0
    segs: List[Tuple[int,int,str]] = []
    if debug:
        print(f"[SEG] {Path(morpheme_path).name}: n={len(rows)}, unit={'ms' if time_scale==1000.0 else 's'}")
    for idx, seg in enumerate(rows):
        attrs = seg.get("attributes") or []
        gloss = (attrs[0].get("name") if attrs and isinstance(attrs[0], dict) else seg.get("gloss") or "UNK")
        s_sec = float(seg.get("start", 0.0)) / time_scale
        e_sec = float(seg.get("end", 0.0)) / time_scale
        s, e = round_frame(s_sec, fps), round_frame(e_sec, fps)
        if debug:
            print(f"  - #{idx:03d} {gloss}: {s_sec:.3f}s→{e_sec:.3f}s | frames {s}→{e} (len={e-s})")
        if e > s:
            segs.append((s, e, str(gloss)))
    return segs

# ---------- convert one video ----------
def convert_one_video(morpheme_path: str, key_dir: Path, out_npz_path: str,
                      fps: int, assume_normalized: bool, width: int, height: int,
                      save_mode: str, debug_segments: bool=False,
                      quiet: bool=False) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,str]]]:
    # 패턴별 rglob로 키파일 수집 (WindowsPath.lower 오류 방지)
    keyfiles: List[str] = []
    for pat in KEY_PATTERNS:
        keyfiles.extend([str(p) for p in key_dir.rglob(pat)])
    keyfiles = sorted(set(keyfiles))
    if not keyfiles:
        raise FileNotFoundError(f"No keypoints under {key_dir}")

    segments = load_morpheme_segments(morpheme_path, fps, debug=debug_segments)
    frames_pts, frames_msk = [], []

    for kf in keyfiles:
        pts = np.zeros((K, 2), np.float32)
        msk = np.zeros((K,), np.uint8)

        jd = safe_load_json(kf)
        ppl = jd.get("people") if isinstance(jd, dict) else None
        if isinstance(ppl, dict):
            ppl = [ppl]
        if isinstance(ppl, list) and len(ppl) > 0:
            try:
                mp = convert_frame_openpose_to_mp(jd, width, height, normalize=not assume_normalized)
                pts, msk = pack_fixed_k(mp)
            except Exception:
                if not quiet:
                    print(f"[WARN] convert failed, zero-padded: {kf}")
        else:
            if not quiet:
                print(f"[WARN] empty/bad json people → zero-padded: {kf}")

        frames_pts.append(pts)
        frames_msk.append(msk)

    kpt = np.stack(frames_pts, axis=0)   # [T,K,2]
    mask = np.stack(frames_msk, axis=0)  # [T,K]
    Path(out_npz_path).parent.mkdir(parents=True, exist_ok=True)
    save_npz(out_npz_path,
             {"kpt": kpt.astype(np.float32), "mask": mask.astype(np.uint8), "fps": np.array(int(fps))},
             mode=save_mode)
    if not quiet:
        print(f"✅ saved {out_npz_path} (T={kpt.shape[0]}, K={kpt.shape[1]}, fps={fps})")
    return kpt, mask, segments

# ---------- templates ----------
def export_templates(kpt: np.ndarray, mask: np.ndarray, segments: List[Tuple[int,int,str]],
                     out_root: Path, min_len: int, max_len: int,
                     verbose_skip: bool=False, long_policy: str="center", chunk_stride: Optional[int]=None,
                     save_mode: str="uncompressed"):
    out_root.mkdir(parents=True, exist_ok=True)
    counters: Dict[str, int] = {}
    total = 0
    T = kpt.shape[0]
    if chunk_stride is None:
        chunk_stride = max(1, max_len // 2)

    for s, e, gloss in segments:
        s = max(0, min(T, s)); e = max(0, min(T, e))
        L = e - s
        if L <= 0:
            if verbose_skip: print(f"[SKIP] {gloss}: empty segment"); continue
        if L < min_len:
            if verbose_skip: print(f"[SKIP] {gloss}: length={L} < min_len={min_len}"); continue

        if L > max_len:
            if long_policy == "skip":
                if verbose_skip: print(f"[SKIP] {gloss}: length={L} > max_len={max_len}"); continue
            elif long_policy == "center":
                mid = (s + e) // 2
                s2 = max(0, mid - max_len // 2)
                e2 = min(T, s2 + max_len)
                parts = [(s2, e2)]
            elif long_policy == "chunk":
                parts = []
                pos = s
                while pos + min_len <= e:
                    s2 = pos
                    e2 = min(e, s2 + max_len)
                    if e2 - s2 >= min_len:
                        parts.append((s2, e2))
                    pos += chunk_stride
            else:
                if verbose_skip: print(f"[SKIP] {gloss}: unknown long_policy={long_policy}"); continue
        else:
            parts = [(s, e)]

        for s2, e2 in parts:
            clip_kpt = kpt[s2:e2]; clip_msk = mask[s2:e2]
            gdir = out_root / gloss
            gdir.mkdir(parents=True, exist_ok=True)
            idx = counters.get(gloss, 0)
            outp = gdir / f"sample_{idx:03d}.npz"
            save_npz(str(outp), {"kpt": clip_kpt.astype(np.float32), "mask": clip_msk.astype(np.uint8)}, mode=save_mode)
            counters[gloss] = idx + 1
            total += 1

    print(f"🧩 exported {total} templates under '{out_root}'")

# ---------- helpers ----------
def templates_exist(tpl_dir: Optional[Path]) -> bool:
    return bool(tpl_dir) and tpl_dir.exists() and any(tpl_dir.rglob("*.npz"))

# ---------- worker ----------
def worker(task):
    mor_path, a = task
    base = mor_path.stem.replace("_morpheme", "")

    # shard paths
    shard_rel = compute_shard(base, a["shard_mode"], a["bucket_size"])
    out_npz = Path(a["out_root"]) / shard_rel / f"{base}.npz"
    tpl_dir = (Path(a["templates_root"]) / shard_rel / base) if a["export_templates"] else None

    # legacy(flat) paths for skip
    legacy_npz = Path(a["out_root"]) / f"{base}.npz"
    legacy_tpl_dir = (Path(a["templates_root"]) / base) if a["export_templates"] else None

    npz_exists = out_npz.exists() or legacy_npz.exists()
    tpl_exists = templates_exist(tpl_dir) or templates_exist(legacy_tpl_dir)

    # (A) 둘 다 있으면 전체 스킵
    if a["skip_existing_npz"] and npz_exists and (not a["export_templates"] or (a["skip_existing_templates"] and tpl_exists)):
        return base, "skipped_all"

    # (B) npz만 있고 템플릿 필요 → npz 로드해서 템플릿만 생성
    if a["export_templates"] and a["skip_existing_npz"] and npz_exists and (not (a["skip_existing_templates"] and tpl_exists)):
        npz_path = out_npz if out_npz.exists() else legacy_npz
        d = np.load(npz_path)
        kpt, mask = d["kpt"], d["mask"]
        export_templates(
            kpt=kpt, mask=mask, segments=load_morpheme_segments(str(mor_path), a["fps"], debug=False),
            out_root=tpl_dir if tpl_dir is not None else legacy_tpl_dir,
            min_len=a["min_len"], max_len=a["max_len"],
            verbose_skip=a["verbose_skip"], long_policy=a["long_policy"], chunk_stride=a["chunk_stride"],
            save_mode=a["save_mode"]
        )
        return base, "templates_from_npz"

    # (C) 변환 필요 → key_dir 찾기(인덱스 → 폴백 검색)
    key_dir = a["__key_index__"].get(base)
    if key_dir is None:
        key_dir = fallback_find_key_dir(a["keypoint_roots"], base)
        if key_dir is None:
            return base, "no_key_dir"

    try:
        kpt, mask, segments = convert_one_video(
            morpheme_path=str(mor_path),
            key_dir=key_dir,
            out_npz_path=str(out_npz),
            fps=a["fps"],
            assume_normalized=a["assume_normalized"],
            width=a["width"], height=a["height"],
            save_mode=a["save_mode"],
            debug_segments=a["debug_segments"],
            quiet=(a["workers"] > 1)
        )
    except Exception as e:
        return base, f"convert_error:{e}"

    if a["export_templates"]:
        if a["skip_existing_templates"] and tpl_exists:
            return base, "npz_ok_tpl_skipped"
        export_templates(
            kpt=kpt, mask=mask, segments=segments,
            out_root=tpl_dir,
            min_len=a["min_len"], max_len=a["max_len"],
            verbose_skip=a["verbose_skip"],
            long_policy=a["long_policy"], chunk_stride=a["chunk_stride"],
            save_mode=a["save_mode"]
        )
        return base, "ok_exported"
    else:
        return base, "ok_npz"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morpheme_root", required=True)
    # 단수/복수 중 하나만 써도 되고, 복수 옵션을 쓰면 ;,: 로 여러 경로 전달
    ap.add_argument("--keypoint_root", default=None)
    ap.add_argument("--keypoint_roots", default=None,
                    help="여러 keypoint root는 ;,: 로 구분 (예: 'data/01;data/02')")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--assume_normalized", action="store_true")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--limit", type=int, default=0)

    # speed/storage
    ap.add_argument("--save_mode", type=str, default="uncompressed", choices=["uncompressed","compressed"])
    ap.add_argument("--workers", type=int, default=1)

    # templates
    ap.add_argument("--export_templates", action="store_true")
    ap.add_argument("--templates_root", type=str, default="data/templates_crowd")
    ap.add_argument("--min_len", type=int, default=18)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--long_policy", type=str, default="center", choices=["skip","center","chunk"])
    ap.add_argument("--chunk_stride", type=int, default=None)

    # debug
    ap.add_argument("--debug_segments", action="store_true")
    ap.add_argument("--verbose_skip", action="store_true")

    # skip
    ap.add_argument("--skip_existing_npz", action="store_true")
    ap.add_argument("--skip_existing_templates", action="store_true")

    # sharding
    ap.add_argument("--shard_mode", type=str, default="fs_sen_bucket",
                    choices=["none","fs_sen","fs_sen_bucket","hash2"])
    ap.add_argument("--bucket_size", type=int, default=1000)

    args = ap.parse_args()

    mor_files = sorted(Path(args.morpheme_root).rglob("*_morpheme.json"))
    print(f"🔍 found morpheme files: {len(mor_files)}")
    if args.limit and args.limit > 0:
        mor_files = mor_files[:args.limit]

    # keypoint roots 집계
    kp_roots: List[str] = []
    if args.keypoint_roots:
        kp_roots = [p.strip() for p in re.split(r"[;,:]", args.keypoint_roots) if p.strip()]
    elif args.keypoint_root:
        kp_roots = [args.keypoint_root]
    else:
        raise ValueError("Either --keypoint_root or --keypoint_roots must be provided.")

    key_index = build_key_index_multi(kp_roots)

    args_dict = {
        "keypoint_roots": kp_roots,
        "out_root": args.out_root,
        "fps": args.fps,
        "assume_normalized": args.assume_normalized,
        "width": args.width, "height": args.height,
        "export_templates": args.export_templates,
        "templates_root": args.templates_root,
        "min_len": args.min_len, "max_len": args.max_len,
        "long_policy": args.long_policy, "chunk_stride": args.chunk_stride,
        "debug_segments": args.debug_segments, "verbose_skip": args.verbose_skip,
        "save_mode": args.save_mode, "workers": args.workers,
        "skip_existing_npz": args.skip_existing_npz,
        "skip_existing_templates": args.skip_existing_templates,
        "shard_mode": args.shard_mode, "bucket_size": args.bucket_size,
        "__key_index__": key_index,
    }

    tasks = [(mor, args_dict) for mor in mor_files]

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        total = len(tasks); done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    base, status = fut.result()
                except Exception as e:
                    base, status = "unknown", f"worker_exception:{e}"
                done += 1
                print(f"[{done}/{total}] {base}: {status}")
    else:
        total = len(tasks)
        for i, t in enumerate(tasks, 1):
            try:
                base, status = worker(t)
            except Exception as e:
                base, status = "unknown", f"worker_exception:{e}"
            print(f"[{i}/{total}] {base}: {status}")

    print("✅ Batch conversion complete.")

if __name__ == "__main__":
    main()
