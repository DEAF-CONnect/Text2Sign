import json
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="원본 manifest 경로")
    parser.add_argument("--out_manifest", required=True, help="줄인 manifest 저장 경로")
    parser.add_argument(
        "--max_per_gloss",
        type=int,
        default=2,
        help="글로스당 최대 템플릿(sample) 개수 (기본 2개)",
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)

    # 1) 구조 체크: meta + index 형태인지
    index = manifest.get("index")
    if not isinstance(index, dict):
        raise RuntimeError(
            "manifest 에서 'index' 키를 찾을 수 없거나 형식이 dict 가 아님.\n"
            f"type(index) = {type(index)}"
        )

    max_n = args.max_per_gloss

    original_total_samples = 0
    new_total_samples = 0
    changed_gloss_count = 0

    # 2) 글로스별 samples 자르기
    for gloss, entry in index.items():
        if not isinstance(entry, dict):
            continue

        samples = entry.get("samples")
        if not isinstance(samples, list):
            continue

        count_before = len(samples)
        original_total_samples += count_before

        if count_before > max_n:
            # 앞에서부터 max_n 개만 유지
            entry["samples"] = samples[:max_n]
            changed_gloss_count += 1

        count_after = len(entry.get("samples", []))
        new_total_samples += count_after

    # 3) meta 업데이트
    meta = manifest.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    meta["shrink_by_gloss"] = True
    meta["max_per_gloss"] = max_n
    meta["original_file_count"] = original_total_samples
    meta["new_file_count"] = new_total_samples
    meta["gloss_count"] = len(index)

    # file_count도 실제 남은 파일 수로 맞춰주기
    meta["file_count"] = new_total_samples

    manifest["meta"] = meta

    # 4) 저장
    save_json(manifest, args.out_manifest)

    print("===== shrink manifest 완료 =====")
    print(f"글로스 수: {len(index)}")
    print(f"원래 총 samples 수: {original_total_samples}")
    print(f"줄인 후 총 samples 수: {new_total_samples}")
    print(f"글로스당 최대 {max_n}개만 유지")
    print(f"실제로 잘린 글로스 수: {changed_gloss_count}")
    print(f"새 manifest 경로: {args.out_manifest}")


if __name__ == "__main__":
    main()
