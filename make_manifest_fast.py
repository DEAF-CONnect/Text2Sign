# make_manifest_fast.py
# ------------------------------------------------------------
# data/templates_xxx/ 구조를 빠르게 스캔해서
# gloss별 파일 목록 + 개수를 JSON으로 저장
# ------------------------------------------------------------
import argparse, json
from pathlib import Path
from datetime import datetime

def make_manifest(templates_root: str, out_path: str):
    root = Path(templates_root)
    if not root.exists():
        raise FileNotFoundError(f"❌ templates_root not found: {root}")

    index = {}
    files = sorted(root.rglob("*.npz"))
    print(f"🔍 scanning {len(files)} npz files under {root}...")

    for i, p in enumerate(files, 1):
        gloss = p.parent.name
        rel = str(p.relative_to(root))
        entry = index.setdefault(gloss, {"samples": [], "count": 0})
        entry["samples"].append({"path": rel})
        entry["count"] += 1
        if i % 5000 == 0:
            print(f"  processed {i} files...")

    manifest = {
        "meta": {
            "templates_root": str(root),
            "file_count": len(files),
            "gloss_count": len(index),
            "created_at": datetime.now().isoformat(timespec="seconds")
        },
        "index": index
    }

    out_path = Path(out_path)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved manifest: {out_path} (glosses={len(index)}, files={len(files)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_root", required=True, help="예: data/templates_realsen_v3")
    ap.add_argument("--out", required=True, help="저장 경로 예: manifest_realsen_v3.json")
    args = ap.parse_args()
    make_manifest(args.templates_root, args.out)


if __name__ == "__main__":
    main()
